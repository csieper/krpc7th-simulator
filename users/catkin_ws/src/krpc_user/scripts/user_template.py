
#!/usr/bin/python3
# -*- coding:utf-8 -*-

import json
import rospy
import actionlib

from platform_msgs.msg import UserLogic
from platform_msgs.srv import StopProcessingUserNode, StopProcessingUserNodeResponse
from geometry_msgs.msg import PoseStamped

from krpc_ib2_msgs.srv import MissionStart, MissionFinished, CheckPoints
from krpc_ib2_msgs.msg import CheckpointPassedAction, CheckpointPassedGoal

from ib2_msgs.msg import CtlCommandAction, CtlCommandGoal
from ib2_msgs.msg import CtlCommandResult, CtlStatusType


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

        self.__subscriber_start = rospy.Subscriber(
            '/ib2_user/start', UserLogic, self.__callback_start
        )

        self.__stop_processing_server = rospy.Service(
            '/ib2_user/stop',
            StopProcessingUserNode,
            self.__stop_processing
        )

        self.__started = False
        self.__action_active = False

        rospy.set_param('/ib2_user/ready', True)

    def __notify_mission_start(self):
        rospy.wait_for_service('/event/mission_start')
        res = self.__mission_start()

        rospy.loginfo(
            'mission_start: success=%s message=%s',
            res.success, res.message
        )

        return res.success

    def __notify_mission_finished(self):
        rospy.wait_for_service('/event/mission_finished')
        res = self.__mission_finished()

        rospy.loginfo(
            'mission_finished: success=%s message=%s',
            res.success, res.message
        )

        return res.success

    def __notify_checkpoint_passed(self, checkpoint_id):
        rospy.loginfo('Passing checkpoint %d', checkpoint_id)

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
                    'checkpoint %d result: %s',
                    checkpoint_id, result
                )

                return result.success

        self.__checkpoint_client.cancel_goal()
        return False

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
                    'Invalid position for checkpoint %d' % checkpoint_id
                )

            if not isinstance(quaternion, list) or len(quaternion) < 4:
                raise ValueError(
                    'Invalid quaternion for checkpoint %d' % checkpoint_id
                )

            targets.append({
                'checkpoint_id': checkpoint_id,
                'position': [float(x) for x in position[:3]],
                'quaternion': [float(x) for x in quaternion[:4]]
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

    def __move_to_checkpoint(self, checkpoint):
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
                if self.__command.wait_for_result(rospy.Duration(0.2)):
                    result = self.__command.get_result()

                    rospy.loginfo(
                        'Checkpoint %d movement result: %s',
                        checkpoint_id,
                        result
                    )

                    return (
                        result.type ==
                        CtlCommandResult.TERMINATE_SUCCESS
                    )

            self.__command.cancel_goal()
            return False

        finally:
            self.__action_active = False

    def __run_user_flow(self):
        try:
            # 1. 開始任務
            if not self.__notify_mission_start():
                return

            # 2. 取得全部檢查點
            checkpoints = self.__resolve_checkpoints()

            # 3. 依序前往並回報
            for checkpoint in checkpoints:

                if rospy.is_shutdown():
                    return

                checkpoint_id = checkpoint['checkpoint_id']

                rospy.loginfo(
                    'Processing checkpoint %d',
                    checkpoint_id
                )

                # 第一次移動
                success = self.__move_to_checkpoint(checkpoint)

                # 第一次失敗，重試一次
                if not success:
                    rospy.logwarn(
                        'Retrying checkpoint %d',
                        checkpoint_id
                    )

                    if rospy.is_shutdown():
                        return

                    success = self.__move_to_checkpoint(checkpoint)

                # 重試仍然失敗
                if not success:
                    rospy.logerr(
                        'Failed to reach checkpoint %d after retry',
                        checkpoint_id
                    )
                    return

                # 回報檢查點
                if not self.__notify_checkpoint_passed(checkpoint_id):
                    rospy.logerr(
                        'Failed to pass checkpoint %d',
                        checkpoint_id
                    )
                    return

            # 4. 全部完成後結束任務
            if not rospy.is_shutdown():
                self.__notify_mission_finished()

        except Exception as exc:
            rospy.logerr('User flow failed: %s', exc)

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
            rospy.logerr('Stop processing failed: %s', exc)

            return StopProcessingUserNodeResponse(
                result=StopProcessingUserNodeResponse.ERROR
            )

    def run(self):
        rospy.spin()


if __name__ == '__main__':
    UserTemplate().run()
