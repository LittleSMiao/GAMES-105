import numpy as np
from scipy.spatial.transform import Rotation as R

def get_from_to_rotation(from_vec, to_vec):
    d = np.dot(from_vec, to_vec)

    if (d < 0.99999):
        a = np.cross(from_vec, to_vec)
        w = d + (d * d + np.dot(a, a)) ** 0.5
        norm = (w * w + a[0] * a[0] + a[1] * a[1] + a[2] * a[2])
        a = a / norm
        w = w / norm

        # rotations.append(R.from_quat([a[0], a[1], a[2], w]))
        return R.from_quat([a[0], a[1], a[2], w])
    else:
        return R.identity()

def part1_inverse_kinematics(meta_data, joint_positions, joint_orientations, target_pose):
    """
    完成函数，计算逆运动学
    输入: 
        meta_data: 为了方便，将一些固定信息进行了打包，见上面的meta_data类
        joint_positions: 当前的关节位置，是一个numpy数组，shape为(M, 3)，M为关节数
        joint_orientations: 当前的关节朝向，是一个numpy数组，shape为(M, 4)，M为关节数
        target_pose: 目标位置，是一个numpy数组，shape为(3,)
    输出:
        经过IK后的姿态
        joint_positions: 计算得到的关节位置，是一个numpy数组，shape为(M, 3)，M为关节数
        joint_orientations: 计算得到的关节朝向，是一个numpy数组，shape为(M, 4)，M为关节数
    """
    path, path_name, path1, path2 = meta_data.get_path_from_root_to_end()

    joint_parent = meta_data.joint_parent
    joint_names = meta_data.joint_name

    chain_joints_positions = [joint_positions[i] for i in path]
    chain_joints_orientations = [R.from_quat(joint_orientations[i]) for i in path]

    joint_initial_position = meta_data.joint_initial_position

    # 保存一下所有节点相对的 offset 和 orientation
    joints_rotation = list()
    joints_offset = list()
    for index, joint in enumerate(joint_names):
        parent = joint_parent[index]
        if (parent == -1):
            joints_offset.append(joint_initial_position[index])
            joints_rotation.append(R.from_quat(joint_orientations[index]))
        else:
            joints_offset.append(joint_initial_position[index] - joint_initial_position[parent])
            cur_orientation = R.from_quat(joint_orientations[index])
            parent_orientation = R.from_quat(joint_orientations[parent])
            joints_rotation.append(parent_orientation.inv() * cur_orientation)

    # 这里不采用雅克比矩阵方法，因为 normal 方向不太确定， 而且normal方向非常可能发生改变
    # 因此在这里欲采用CCD的方法

    cur_index = len(path) - 4
    rot_joints_len = len(path) - 1
    cnt = 0

    while (cur_index < 0):
        cur_index += rot_joints_len

    dist = np.linalg.norm(target_pose - chain_joints_positions[-1])
    while dist > 0.01 and cnt < 15:
        cur_end_pos = chain_joints_positions[-1]
        cur_joint_pos = chain_joints_positions[cur_index]

        rotation = get_from_to_rotation(cur_end_pos - cur_joint_pos, target_pose - cur_joint_pos)
        chain_joints_orientations[cur_index] = rotation * chain_joints_orientations[cur_index]

        pre_pos = chain_joints_positions[cur_index]

        for f_index in range(cur_index + 1, len(path)):
            ori_pos = chain_joints_positions[f_index]
            f_offset = chain_joints_positions[f_index] - cur_joint_pos
            chain_joints_positions[f_index] = cur_joint_pos + rotation.apply(f_offset)
            r = get_from_to_rotation(ori_pos - pre_pos, chain_joints_positions[f_index] - chain_joints_positions[f_index - 1])
            if f_index > cur_index + 1:
                chain_joints_orientations[f_index - 1] = r * chain_joints_orientations[f_index - 1]
            pre_pos = ori_pos

        chain_joints_orientations[-1] = chain_joints_orientations[-2]

        dist = np.linalg.norm(target_pose - chain_joints_positions[-1])

        cur_index -= 1
        if (cur_index < 0):
            cur_index += rot_joints_len
        cnt += 1

    for i in range(len(path2)):
        chain_cur_index = len(path2) - 1 - i
        cur_index = path2[chain_cur_index]
        parent = joint_parent[cur_index]

        if parent != -1 and i > 1:
            chain_joints_orientations[chain_cur_index + 1] = get_from_to_rotation(joints_offset[cur_index], chain_joints_positions[chain_cur_index] - chain_joints_positions[chain_cur_index + 1])

    for i in range(len(joint_names)):
        parent = joint_parent[i]
        # 如果在 path 内，则选用 IK 解算出来的世界坐标
        # 否则采用与父节点的相对坐标解算出来世界坐标
        # 从前到后遍历的时候，保证解算子节点的时候，父节点的朝向和位置已经解算完毕
        if (i in path):
            chain_index = path.index(i)
            joint_positions[i] = chain_joints_positions[chain_index]
            joint_orientations[i] = chain_joints_orientations[chain_index].as_quat()
        else:
            if (parent != -1):
                p_rot = R.from_quat(joint_orientations[parent])
                joint_orientations[i] = (p_rot * joints_rotation[i]).as_quat()
                joint_positions[i] = joint_positions[parent] + p_rot.apply(joints_offset[i])

    return joint_positions, joint_orientations

def part2_inverse_kinematics(meta_data, joint_positions, joint_orientations, relative_x, relative_z, target_height):
    """
    输入lWrist相对于RootJoint前进方向的xz偏移，以及目标高度，IK以外的部分与bvh一致
    """
    # 其他节点已经解算完毕了
    target_global = np.array([joint_positions[0][0] + relative_x, target_height, joint_positions[0][2] + relative_z])
    joint_positions, joint_orientations = part1_inverse_kinematics(meta_data, joint_positions, joint_orientations, target_global)

    return joint_positions, joint_orientations

def bonus_inverse_kinematics(meta_data, joint_positions, joint_orientations, left_target_pose, right_target_pose):
    """
    输入左手和右手的目标位置，固定左脚，完成函数，计算逆运动学
    """

    meta_data.root_joint = 'lToeJoint_end'
    meta_data.end_joint = 'lWrist_end'
    joint_positions, joint_orientations = part1_inverse_kinematics(meta_data, joint_positions, joint_orientations,
                                                                   left_target_pose)

    meta_data.root_joint = 'rShoulder'
    meta_data.end_joint = 'rWrist_end'
    joint_positions, joint_orientations = part1_inverse_kinematics(meta_data, joint_positions, joint_orientations,
                                                                   right_target_pose)

    return joint_positions, joint_orientations


if __name__ == '__main__':
    a = np.array([3, 4])
    print(np.linalg.norm(a))
    pass