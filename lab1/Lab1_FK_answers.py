import numpy as np
from scipy.spatial.transform import Rotation as R

def part1_calculate_T_pose(bvh_file_path):
    """请填写以下内容
    输入： bvh 文件路径
    输出:
        joint_name: List[str]，字符串列表，包含着所有关节的名字
        joint_parent: List[int]，整数列表，包含着所有关节的父关节的索引,根节点的父关节索引为-1
        joint_offset: np.ndarray，形状为(M, 3)的numpy数组，包含着所有关节的偏移量

    Tips:
        joint_name顺序应该和bvh一致
    """

    cur_indices = list()
    joint_name = list()
    joint_parent = list()
    joint_offsets = list()

    cur_index = 0

    with open(bvh_file_path)  as f:
        for line in f.readlines():
            entries = line.split()
            if (not entries):
                continue

            type = entries[0]

            if (type == 'HIERARCHY'):
                pass
            elif (type == 'OFFSET'):
                joint_offsets.append(np.array([float(entries[1]), float(entries[2]), float(entries[3])]))
            elif (type == 'CHANNELS'):
                pass
            elif (type == 'JOINT'):
                cur_index += 1
                joint_parent.append(cur_indices[-1])
                joint_name.append(entries[1])
            elif (type == '{'):
                cur_indices.append(cur_index)
                pass
            elif (type == '}'):
                cur_indices.pop()
                pass
            elif (type == 'End'):
                pindex = cur_indices[-1]
                pname = joint_name[pindex]
                node_name = pname + '_end'

                cur_index += 1
                joint_parent.append(pindex)
                joint_name.append(node_name)
            elif (type == 'ROOT'):
                node_name = entries[1]
                joint_name.append(node_name)
                joint_parent.append(-1)
                cur_index = 0
            elif (type == 'MOTION'):
                break

    joint_offsets = np.array(joint_offsets)
    return joint_name, joint_parent, joint_offsets


def load_motion_data(bvh_file_path):
    """part2 辅助函数，读取bvh文件"""
    with open(bvh_file_path, 'r') as f:
        lines = f.readlines()
        for i in range(len(lines)):
            if lines[i].startswith('Frame Time'):
                break
        motion_data = []
        for line in lines[i+1:]:
            data = [float(x) for x in line.split()]
            if len(data) == 0:
                break
            motion_data.append(np.array(data).reshape(1,-1))
        motion_data = np.concatenate(motion_data, axis=0)
    return motion_data


def part2_forward_kinematics(joint_name, joint_parent, joint_offset, motion_data, frame_id):
    """请填写以下内容
    输入: part1 获得的关节名字，父节点列表，偏移量列表
        motion_data: np.ndarray，形状为(N,X)的numpy数组，其中N为帧数，X为Channel数
        frame_id: int，需要返回的帧的索引
    输出:
        joint_positions: np.ndarray，形状为(M, 3)的numpy数组，包含着所有关节的全局位置
        joint_orientations: np.ndarray，形状为(M, 4)的numpy数组，包含着所有关节的全局旋转(四元数)
    Tips:
        1. joint_orientations的四元数顺序为(x, y, z, w)
    """
    data = motion_data[frame_id, :]
    joint_positions = list()
    joint_orientations = list()

    joint_positions.append(data[0 : 3].reshape(1, -1))
    eulers = data[3: 6]
    rotation = R.from_euler('XYZ', eulers, degrees=True)
    joint_orientations.append(rotation)
    index = 6
    for i, joint in enumerate(joint_name):
        pindex = joint_parent[i]
        if (pindex == -1):
            continue
        if joint.endswith('_end'):
            joint_orientations.append(joint_orientations[pindex])
            pos = joint_positions[pindex] + joint_orientations[pindex].apply(joint_offset[i])
            joint_positions.append(pos)
        else:
            eulers = data[index : index + 3]
            rotation = R.from_euler('XYZ', eulers, degrees=True)
            g_rotation = joint_orientations[pindex] * rotation
            joint_orientations.append(g_rotation)

            pos = joint_positions[pindex] + joint_orientations[pindex].apply(joint_offset[i])
            joint_positions.append(pos)
            index += 3
    joint_positions = np.concatenate(joint_positions, axis=0)
    joint_orientations = [r.as_quat() for r in joint_orientations]
    joint_orientations = np.array(joint_orientations)

    return joint_positions, joint_orientations


def part3_retarget_func(T_pose_bvh_path, A_pose_bvh_path):
    """
    将 A-pose的bvh重定向到T-pose上
    输入: 两个bvh文件的路径
    输出: 
        motion_data: np.ndarray，形状为(N,X)的numpy数组，其中N为帧数，X为Channel数。retarget后的运动数据
    Tips:
        两个bvh的joint name顺序可能不一致哦(
    """

    t_pose_joint_name, t_pose_joint_parent, t_pose_joint_offset = part1_calculate_T_pose(T_pose_bvh_path)
    a_pose_joint_name, a_pose_joint_parent, a_pose_joint_offset = part1_calculate_T_pose(A_pose_bvh_path)

    # 需要计算 t_pose 和 a_pose 初始状态下世界坐标系的位置
    t_global_pose = list()
    a_global_pose = list()

    motion_data_index = dict()

    index = 0
    # 在 motion_data 中的索引
    for i, joint in enumerate(a_pose_joint_name):
        if joint.endswith('_end'):
            continue
        motion_data_index[joint] = index
        index += 1
        pass

    for i, joint in enumerate(t_pose_joint_name):
        pindex = t_pose_joint_parent[i]
        if (pindex == -1):
            t_global_pose.append(t_pose_joint_offset[i])
        else:
            t_global_pose.append(t_global_pose[pindex] + t_pose_joint_offset[i])

    indices = dict()
    for i, joint in enumerate(a_pose_joint_name):
        pindex = a_pose_joint_parent[i]

        indices[joint] = i
        if (pindex == -1):
            a_global_pose.append(a_pose_joint_offset[i])
        else:
            a_global_pose.append(a_global_pose[pindex] + a_pose_joint_offset[i])

    rotations = [R.identity() for _ in range(len(t_pose_joint_name))]
    p_rotations = [R.identity() for _ in range(len(t_pose_joint_name))]

    # 这里应该是自己到儿子的
    for i, joint in enumerate(t_pose_joint_name):
        pindex = t_pose_joint_parent[i]
        if pindex == -1:
            rotations[i] = R.identity()
            p_rotations[i] = R.identity()
            continue
        if joint.endswith('_end'):
            # rotations.append(R.identity())
            rotations[i] = R.identity()
            # continue

        index_a = indices[joint]
        pindex_a = a_pose_joint_parent[index_a]

        from_vec = a_global_pose[index_a] - a_global_pose[pindex_a]
        to_vec = t_global_pose[i] - t_global_pose[pindex]

        d = np.dot(from_vec, to_vec)
        if (d < 0.99999):
            a = np.cross(from_vec, to_vec)
            w = d + (d * d + np.dot(a, a)) ** 0.5
            norm = (w * w + a[0] * a[0] + a[1] * a[1] + a[2] * a[2])
            a = a / norm
            w = w / norm

            # rotations.append(R.from_quat([a[0], a[1], a[2], w]))
            rotations[pindex] = R.from_quat([a[0], a[1], a[2], w])
        else:
            rotations[pindex] = R.identity()

        from_vec = a_global_pose[pindex_a] - a_global_pose[index_a]
        to_vec = t_global_pose[pindex] - t_global_pose[i]

        d = np.dot(from_vec, to_vec)
        if (d < 0.99999):
            a = np.cross(from_vec, to_vec)
            w = d + (d * d + np.dot(a, a)) ** 0.5
            norm = (w * w + a[0] * a[0] + a[1] * a[1] + a[2] * a[2])
            a = a / norm
            w = w / norm

            # rotations.append(R.from_quat([a[0], a[1], a[2], w]))
            p_rotations[i] = R.from_quat([a[0], a[1], a[2], w])
        else:
            p_rotations[i] = R.identity()


    a_pose_motion_data = load_motion_data(A_pose_bvh_path)
    motion_data = a_pose_motion_data[:, 0 : 6]

    # 在这里还存在的一个问题是因为 _end 节点的存在，从 eulers 中取数的问题

    for i, joint in enumerate(t_pose_joint_name):
        pindex = t_pose_joint_parent[i]
        if (pindex == -1):
            continue
        if joint.endswith('_end'):
            continue

        index_a = motion_data_index[joint]
        eulers = a_pose_motion_data[:, 3 + index_a * 3 : 6 + index_a * 3]

        r = R.from_euler('XYZ', eulers, degrees=True)
        r = p_rotations[i] * r * rotations[i].inv()
        r = r.as_euler('XYZ', degrees=True)

        motion_data = np.append(motion_data, r, axis=1)

    return motion_data


if __name__ == '__main__':
    t_pose_bvh_file = 'data/walk60.bvh'
    a_pose_bvh_file = 'data/A_pose_run.bvh'

    m_data = part3_retarget_func(t_pose_bvh_file, a_pose_bvh_file)
    print(m_data.shape)
