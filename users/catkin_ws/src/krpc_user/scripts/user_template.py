
#!/usr/bin/python3
# -*- coding:utf-8 -*-

import json
import rospy
import actionlib
import itertools
import math
import time
import threading

from platform_msgs.msg import UserLogic
from platform_msgs.srv import StopProcessingUserNode, StopProcessingUserNodeResponse
from geometry_msgs.msg import PoseStamped

from krpc_ib2_msgs.srv import MissionStart, MissionFinished, CheckPoints
from krpc_ib2_msgs.msg import CheckpointPassedAction, CheckpointPassedGoal

from ib2_msgs.msg import (
    CtlCommandAction,
    CtlCommandGoal,
    CtlCommandResult,
    CtlStatusType,
    Navigation
)


START_POSITION = [
    2.9963087339393,
    -0.5192473538539125,
    0.8725719492809979
]

FINAL_CHECKPOINT_ID = 12

# 導航恢復設定
NAV_TIMEOUT = 1.0
NAV_RECOVERY_TIMEOUT = 5.0
NAV_REQUIRED_SAMPLES = 3
NAV_RETRY_LIMIT = 2

# 目標四元數允許的範圍
QUATERNION_MIN_NORM = 1e-6


class UserTemplate:

    def __init__(self):
        rospy.init_node('krpc_user')

        self.__command = actionlib.SimpleActionClient(
            '/ctl/command', CtlCommandAction
        )

        self.__checkpoint_client = actionlib.SimpleActionClient(
            '/event/checkpoint_passed', CheckpointPassedAction
        )

        self.__mission_start = rospy.ServiceProxy(
            '/event/mission_start', MissionStart
        )

        self.__mission_finished = rospy.ServiceProxy(
            '/event/mission_finished', MissionFinished
        )

        self.__checkpoints = rospy.ServiceProxy(
            '/mission/get_checkpoints', CheckPoints
        )

        # 導航資料監測
        self.__nav_lock = threading.Lock()
        self.__nav_sequence = 0
        self.__nav_valid_count = 0
        self.__nav_last_receive = 0.0
        self.__nav_data = None

        self.__nav_subscriber = rospy.Subscriber(
            '/sensor_fusion/navigation',
            Navigation,
            self.__callback_navigation,
            queue_size=10
        )

        self.__subscriber_start = rospy.Subscriber(
            '/ib2_user/start',
            UserLogic,
            self.__callback_start
        )

        self.__stop_processing_server = rospy.Service(
            '/ib2_user/stop',
            StopProcessingUserNode,
            self.__stop_processing
        )

        self.__started = False
        self.__action_active = False

        rospy.set_param('/ib2_user/ready', True)

    # --------------------------------------------------
    # Navigation
    # --------------------------------------------------

    @staticmethod
    def __finite(value):
        try:
            return math.isfinite(float(value))
        except (TypeError, ValueError):
            return False

    @staticmethod
    def __normalize_quaternion(q):
        if len(q) != 4:
            return None

        values = [float(v) for v in q]

        if not all(math.isfinite(v) for v in values):
            return None

        norm = math.sqrt(sum(v * v for v in values))

        if norm < QUATERNION_MIN_NORM:
            return None

        return [v / norm for v in values]

    def __extract_navigation_pose(self, msg):
        """
        支援常見 ROS pose 包裝：
        PoseStamped
        PoseWithCovarianceStamped
        Navigation.pose.pose
        """

        pose_container = getattr(msg, 'pose', None)

        if pose_container is None:
            return None

        pose = pose_container

        # PoseWithCovarianceStamped -> PoseWithCovariance -> Pose
        for _ in range(2):
            nested = getattr(pose, 'pose', None)

            if nested is None:
                break

            pose = nested

        position = getattr(pose, 'position', None)
        orientation = getattr(pose, 'orientation', None)

        if position is None or orientation is None:
            return None

        p = [
            float(position.x),
            float(position.y),
            float(position.z)
        ]

        q = [
            float(orientation.x),
            float(orientation.y),
            float(orientation.z),
            float(orientation.w)
        ]

        if not all(self.__finite(v) for v in p):
            return None

        q = self.__normalize_quaternion(q)

        if q is None:
            return None

        header = getattr(pose_container, 'header', None)
        frame_id = getattr(header, 'frame_id', '') if header else ''

        return {
            'position': p,
            'quaternion': q,
            'frame_id': frame_id
        }

    def __callback_navigation(self, msg):
        now = time.time()

        try:
            pose_data = self.__extract_navigation_pose(msg)
        except Exception:
            pose_data = None

        with self.__nav_lock:
            self.__nav_sequence += 1
            self.__nav_last_receive = now

            if pose_data is None:
                self.__nav_valid_count = 0
                self.__nav_data = None
                return

            self.__nav_valid_count += 1

            self.__nav_data = {
                'position': pose_data['position'],
                'quaternion': pose_data['quaternion'],
                'frame_id': pose_data['frame_id'],
                'sequence': self.__nav_sequence,
                'receive_time': now
            }

    def __get_navigation_state(self):
        with self.__nav_lock:
            if self.__nav_data is None:
                return None

            return dict(self.__nav_data)

    def __get_navigation_sequence(self):
        with self.__nav_lock:
            return self.__nav_sequence

    def __wait_navigation_ready(
        self,
        after_sequence=None,
        timeout=NAV_RECOVERY_TIMEOUT
    ):
        """
        等待連續收到有效導航資料。
        after_sequence 用來確保 INVALID_NAV 後收到的是新資料。
        """

        if after_sequence is None:
            after_sequence = 0

        start = time.time()

        rate = 0.05

        while not rospy.is_shutdown():
            now = time.time()

            with self.__nav_lock:
                sequence = self.__nav_sequence
                valid_count = self.__nav_valid_count
                last_receive = self.__nav_last_receive
                nav_data = self.__nav_data

            fresh = (
                nav_data is not None and
                last_receive > 0.0 and
                now - last_receive <= NAV_TIMEOUT
            )

            new_data = sequence > after_sequence

            if (
                fresh and
                new_data and
                valid_count >= NAV_REQUIRED_SAMPLES
            ):
                rospy.loginfo(
                    '[NAV] Ready: sequence=%d position=%s',
                    sequence,
                    nav_data['position']
                )
                return True

            if now - start >= timeout:
                rospy.logerr(
                    '[NAV] Recovery timeout: sequence=%d '
                    'valid_count=%d fresh=%s',
                    sequence,
                    valid_count,
                    fresh
                )
                return False

            time.sleep(rate)

        return False

    def __log_navigation_state(self, reason):
        nav = self.__get_navigation_state()

        if nav is None:
            rospy.logerr(
                '[NAV] %s: no valid navigation data',
                reason
            )
            return

        age = time.time() - nav['receive_time']

        rospy.logerr(
            '[NAV] %s: sequence=%d age=%.3f position=%s quaternion=%s',
            reason,
            nav['sequence'],
            age,
            nav['position'],
            nav['quaternion']
        )

    # --------------------------------------------------
    # Mission services
    # --------------------------------------------------

    def __notify_mission_start(self):
        rospy.wait_for_service('/event/mission_start')

        res = self.__mission_start()

        rospy.loginfo(
            'mission_start: success=%s message=%s',
            res.success,
            res.message
        )

        return res.success

    def __notify_mission_finished(self):
        rospy.wait_for_service('/event/mission_finished')

        res = self.__mission_finished()

        rospy.loginfo(
            'mission_finished: success=%s message=%s',
            res.success,
            res.message
        )

        return res.success

    def __notify_checkpoint_passed(self, checkpoint_id):
        notify_start_time = time.time()

        rospy.loginfo(
            'Passing checkpoint %d',
            checkpoint_id
        )

        self.__checkpoint_client.wait_for_server()

        goal = CheckpointPassedGoal()
        goal.checkpoint_id = checkpoint_id

        self.__checkpoint_client.send_goal(goal)

        while not rospy.is_shutdown():
            if self.__checkpoint_client.wait_for_result(
                rospy.Duration(0.2)
            ):
                result = self.__checkpoint_client.get_result()

                rospy.loginfo(
                    '[TIME] checkpoint=%d notification=%.2f sec',
                    checkpoint_id,
                    time.time() - notify_start_time
                )

                rospy.loginfo(
                    'checkpoint %d result: %s',
                    checkpoint_id,
                    result
                )

                return result.success

        self.__checkpoint_client.cancel_goal()
        return False

    # --------------------------------------------------
    # Checkpoints
    # --------------------------------------------------

    def __get_checkpoint_id(self, checkpoint):
        for key in ('checkpoint_id', 'checkpointid', 'id'):
            if key in checkpoint:
                return int(checkpoint[key])

        return None

    def __resolve_checkpoints(self):
        rospy.wait_for_service('/mission/get_checkpoints')

        res = self.__checkpoints()

        rospy.loginfo(
            'get_checkpoints: success=%s',
            res.success
        )

        if not res.success:
            raise RuntimeError('Failed to get checkpoints')

        payload = json.loads(res.message)

        checkpoints = payload.get(
            'checkpoints',
            payload.get('Checkpoints', [])
        )

        targets = []

        for cp in checkpoints:
            if not isinstance(cp, dict):
                continue

            checkpoint_id = self.__get_checkpoint_id(cp)

            if checkpoint_id is None:
                continue

            position = cp.get('position')
            attitude = cp.get('attitude', {})
            quaternion = attitude.get('quaternion')

            if not isinstance(position, list) or len(position) < 3:
                raise ValueError(
                    'Invalid position for checkpoint %d' %
                    checkpoint_id
                )

            if not isinstance(quaternion, list) or len(quaternion) < 4:
                raise ValueError(
                    'Invalid quaternion for checkpoint %d' %
                    checkpoint_id
                )

            p = [float(x) for x in position[:3]]
            q = self.__normalize_quaternion(quaternion[:4])

            if not all(self.__finite(v) for v in p):
                raise ValueError(
                    'Non-finite position for checkpoint %d' %
                    checkpoint_id
                )

            if q is None:
                raise ValueError(
                    'Invalid quaternion for checkpoint %d' %
                    checkpoint_id
                )

            targets.append({
                'checkpoint_id': checkpoint_id,
                'position': p,
                'quaternion': q
            })

        if not targets:
            raise RuntimeError('No valid checkpoints found')

        targets.sort(key=lambda x: x['checkpoint_id'])

        rospy.loginfo(
            'Found %d checkpoints: %s',
            len(targets),
            [x['checkpoint_id'] for x in targets]
        )

        return targets

    # --------------------------------------------------
    # Route optimization
    # --------------------------------------------------

    def __distance(self, a, b):
        return math.sqrt(
            (a[0] - b[0]) ** 2 +
            (a[1] - b[1]) ** 2 +
            (a[2] - b[2]) ** 2
        )

    def __optimize_checkpoint_order(self, checkpoints):
        final_checkpoint = None
        others = []

        for cp in checkpoints:
            if cp['checkpoint_id'] == FINAL_CHECKPOINT_ID:
                final_checkpoint = cp
            else:
                others.append(cp)

        if final_checkpoint is None:
            raise RuntimeError(
                'Final checkpoint %d not found' %
                FINAL_CHECKPOINT_ID
            )

        if len(others) != 7:
            raise RuntimeError(
                'Expected 7 checkpoints before CP12, found %d' %
                len(others)
            )

        best_route = None
        best_distance = float('inf')

        for route in itertools.permutations(others):
            total_distance = 0.0
            current_position = START_POSITION

            for cp in route:
                total_distance += self.__distance(
                    current_position,
                    cp['position']
                )
                current_position = cp['position']

            total_distance += self.__distance(
                current_position,
                final_checkpoint['position']
            )

            if total_distance < best_distance:
                best_distance = total_distance
                best_route = route

        optimized_route = list(best_route)
        optimized_route.append(final_checkpoint)

        rospy.loginfo(
            'Route: %s',
            [cp['checkpoint_id'] for cp in optimized_route]
        )

        rospy.loginfo(
            'Estimated distance: %.4f',
            best_distance
        )

        return optimized_route

    # --------------------------------------------------
    # Movement
    # --------------------------------------------------

    def __move_to_checkpoint(self, checkpoint):
        move_start_time = time.time()
        position = checkpoint['position']
        quaternion = checkpoint['quaternion']
        checkpoint_id = checkpoint['checkpoint_id']

        rospy.loginfo(
            'Moving to checkpoint %d: position=%s quaternion=%s',
            checkpoint_id,
            position,
            quaternion
        )

        target = PoseStamped()
        target.header.frame_id = 'iss_body'

        target.pose.position.x = position[0]
        target.pose.position.y = position[1]
        target.pose.position.z = position[2]

        target.pose.orientation.x = quaternion[0]
        target.pose.orientation.y = quaternion[1]
        target.pose.orientation.z = quaternion[2]
        target.pose.orientation.w = quaternion[3]

        command_type = CtlStatusType(
            type=CtlStatusType.MOVE_TO_ABSOLUTE_TARGET
        )

        goal = CtlCommandGoal(
            target=target,
            type=command_type
        )

        self.__action_active = True

        try:
            self.__command.send_goal(goal)

            while not rospy.is_shutdown():
                if self.__command.wait_for_result(
                    rospy.Duration(0.2)
                ):
                    result = self.__command.get_result()

                    if result is None:
                        rospy.logerr(
                            'Checkpoint %d returned empty result',
                            checkpoint_id
                        )
                        return False, None

                    result_type = result.type

                    rospy.loginfo(
                        '[TIME] checkpoint=%d movement=%.2f sec',
                        checkpoint_id,
                        time.time() - move_start_time
                    )

                    rospy.loginfo(
                        'Checkpoint %d movement result: %s',
                        checkpoint_id,
                        result
                    )

                    return (
                        result_type == CtlCommandResult.TERMINATE_SUCCESS,
                        result_type
                    )

            self.__command.cancel_goal()
            return False, None

        finally:
            self.__action_active = False

    def __move_with_recovery(self, checkpoint):
        checkpoint_id = checkpoint['checkpoint_id']

        # 開始移動前，先確認導航資料有效
        if not self.__wait_navigation_ready():
            self.__log_navigation_state(
                'Navigation unavailable before checkpoint %d' %
                checkpoint_id
            )
            return False

        nav_state = self.__get_navigation_state()
        if nav_state is None:
            rospy.logerr('[NAV] No navigation data before movement')
            return False

        nav_frame = nav_state.get('frame_id', '')
        rospy.loginfo('[NAV] frame_id=%s', nav_frame)

        if nav_frame != 'iss_body':
            rospy.logerr(
                '[NAV] Frame mismatch: navigation=%s target=iss_body. Stop movement.',
                nav_frame
            )
            return False

        for attempt in range(NAV_RETRY_LIMIT + 1):
            if rospy.is_shutdown():
                return False

            rospy.loginfo(
                '[MOVE] checkpoint=%d attempt=%d',
                checkpoint_id,
                attempt + 1
            )

            success, result_type = self.__move_to_checkpoint(
                checkpoint
            )

            if success:
                return True

            rospy.logerr(
                '[MOVE] checkpoint=%d failed with result=%s',
                checkpoint_id,
                result_type
            )
            return False

        return False


    def __run_user_flow(self):
        try:
            if not self.__notify_mission_start():
                return

            # 等待導航資料穩定
            rospy.loginfo('[NAV] Waiting for initial navigation data')

            if not self.__wait_navigation_ready():
                rospy.logerr(
                    '[NAV] Initial navigation unavailable; abort mission'
                )
                return

            checkpoints = self.__resolve_checkpoints()

            checkpoints = self.__optimize_checkpoint_order(
                checkpoints
            )

            for checkpoint in checkpoints:
                if rospy.is_shutdown():
                    return

                checkpoint_id = checkpoint['checkpoint_id']

                rospy.loginfo(
                    'Processing checkpoint %d',
                    checkpoint_id
                )

                success = self.__move_with_recovery(checkpoint)

                if not success:
                    rospy.logerr(
                        'Failed to reach checkpoint %d',
                        checkpoint_id
                    )
                    return

                # 抵達後才回報，不提前開始下一段移動
                if not self.__notify_checkpoint_passed(checkpoint_id):
                    rospy.logerr(
                        'Failed to pass checkpoint %d',
                        checkpoint_id
                    )
                    return

            if not rospy.is_shutdown():
                self.__notify_mission_finished()

        except Exception as exc:
            rospy.logerr(
                'User flow failed: %s',
                exc
            )

    # --------------------------------------------------
    # ROS callbacks
    # --------------------------------------------------

    def __callback_start(self, _msg):
        if self.__started:
            rospy.logwarn('Start callback already processed')
            return

        self.__started = True

        rospy.loginfo('Mission started')
        self.__run_user_flow()

    def __stop_processing(self, _request):
        try:
            if self.__action_active:
                self.__command.cancel_goal()

            rospy.loginfo('Finish processing')

            return StopProcessingUserNodeResponse(
                result=StopProcessingUserNodeResponse.SUCCESS
            )

        except Exception as exc:
            rospy.logerr(
                'Stop processing failed: %s',
                exc
            )

            return StopProcessingUserNodeResponse(
                result=StopProcessingUserNodeResponse.ERROR
            )

    def run(self):
        rospy.spin()


if __name__ == '__main__':
    UserTemplate().run()
