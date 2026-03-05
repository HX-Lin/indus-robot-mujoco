"""
MR2010机械臂逆运动学求解器
使用阻尼最小二乘法 (Damped Least Squares / Levenberg-Marquardt)
"""

import numpy as np
import mujoco

from mr2010_robot import NUM_JOINTS


class DampedLeastSquaresIK:
    """阻尼最小二乘逆运动学求解器"""

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData,
                 site_name: str = "tool_tip", damping: float = 0.01):
        self.model = model
        self.data = data
        self.site_id = model.site(site_name).id
        self.damping = damping
        self.n_joints = NUM_JOINTS
        # Pre-allocate Jacobian arrays
        self.jac_pos = np.zeros((3, model.nv))  # position Jacobian
        self.jac_rot = np.zeros((3, model.nv))  # rotation Jacobian

    def get_ee_pos(self) -> np.ndarray:
        """获取末端执行器当前位置"""
        return self.data.site_xpos[self.site_id].copy()

    def get_ee_mat(self) -> np.ndarray:
        """获取末端执行器当前旋转矩阵 (3x3)"""
        return self.data.site_xmat[self.site_id].reshape(3, 3).copy()

    def solve(self, target_pos: np.ndarray, target_mat: np.ndarray = None,
              max_iter: int = 200, tol: float = 1e-4) -> np.ndarray:
        """
        求解逆运动学

        Args:
            target_pos: 目标位置 (3,)
            target_mat: 目标旋转矩阵 (3,3)，如果为None则只做位置IK
            max_iter: 最大迭代次数
            tol: 收敛阈值

        Returns:
            关节角度 (NUM_JOINTS,)
        """
        damping = self.damping

        for i in range(max_iter):
            mujoco.mj_forward(self.model, self.data)

            # Compute position error
            pos_error = target_pos - self.get_ee_pos()

            if target_mat is not None:
                # Compute orientation error using rotation matrix difference
                current_mat = self.get_ee_mat()
                error_mat = target_mat @ current_mat.T
                # Convert to axis-angle (approximate for small angles)
                rot_error = np.array([
                    error_mat[2, 1] - error_mat[1, 2],
                    error_mat[0, 2] - error_mat[2, 0],
                    error_mat[1, 0] - error_mat[0, 1],
                ]) * 0.5
                error = np.concatenate([pos_error, rot_error])
            else:
                error = pos_error

            # Check convergence
            err_norm = np.linalg.norm(pos_error)
            if err_norm < tol:
                break

            # Adaptive damping: increase when error is large, decrease when small
            damping = self.damping * max(1.0, err_norm * 10)

            # Compute Jacobian
            mujoco.mj_jacSite(self.model, self.data, self.jac_pos, self.jac_rot, self.site_id)

            if target_mat is not None:
                jac = np.vstack([self.jac_pos[:, :self.n_joints],
                                 self.jac_rot[:, :self.n_joints]])  # (6, n_joints)
            else:
                jac = self.jac_pos[:, :self.n_joints]  # (3, n_joints)

            # Damped least squares: dq = J^T (J J^T + lambda^2 I)^{-1} * error
            jjt = jac @ jac.T
            damped = jjt + (damping ** 2) * np.eye(jjt.shape[0])
            dq = jac.T @ np.linalg.solve(damped, error)

            # Limit step size to avoid overshooting
            step_norm = np.linalg.norm(dq)
            max_step = 0.5
            if step_norm > max_step:
                dq = dq * (max_step / step_norm)

            # Apply joint update
            self.data.qpos[:self.n_joints] += dq

            # Clip to joint limits with margin to avoid getting stuck
            margin = 0.01  # ~0.57 degrees
            for j in range(self.n_joints):
                lo = self.model.jnt_range[j, 0]
                hi = self.model.jnt_range[j, 1]
                if lo < hi:  # limits are defined
                    self.data.qpos[j] = np.clip(self.data.qpos[j], lo + margin, hi - margin)

        mujoco.mj_forward(self.model, self.data)
        return self.data.qpos[:self.n_joints].copy()


if __name__ == "__main__":
    from mr2010_robot import create_base_scene

    # Create scene and compile model
    spec = create_base_scene(with_platform=True)
    model = spec.compile()
    data = mujoco.MjData(model)

    # Create IK solver
    ik = DampedLeastSquaresIK(model, data)

    # Test targets within the robot reachable workspace
    # These are positions near the platform surface area
    test_targets = [
        np.array([0.5, 0.0, 0.5]),
        np.array([0.6, 0.0, 0.4]),
        np.array([0.5, -0.1, 0.4]),
        np.array([0.6, 0.1, 0.3]),
    ]

    print("=" * 60)
    print("MR2010 IK Solver Test (Damped Least Squares)")
    print("=" * 60)

    all_passed = True
    pass_threshold = 0.01

    for idx, target in enumerate(test_targets):
        # Reset to home position before each test
        data.qpos[:] = 0.0
        mujoco.mj_forward(model, data)

        result_q = ik.solve(target)
        achieved_pos = ik.get_ee_pos()
        pos_error = np.linalg.norm(target - achieved_pos)
        passed = pos_error < pass_threshold

        if not passed:
            all_passed = False

        status = "PASS" if passed else "FAIL"
        print(f"\nTest {idx + 1}: target={target}")
        print(f"  achieved={achieved_pos}")
        print(f"  joints={np.round(np.degrees(result_q), 1)} deg")
        print(f"  pos_error={pos_error:.6f} m  [{status}]")

    print("\n" + "=" * 60)
    if all_passed:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED")
    print("=" * 60)
