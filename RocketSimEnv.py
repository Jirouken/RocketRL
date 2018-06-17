# -*- coding: utf-8 -*-
import gym
import gym.spaces
from gym.envs.classic_control import rendering
import numpy as np
from scipy.constants import g
import pyglet
from ModelRocketSimulator.quaternion import Quaternion
from ModelRocketSimulator.mass import Mass
from ModelRocketSimulator.force import Force
from ModelRocketSimulator.wind import Wind
import ModelRocketSimulator.angle_of_attack as aoa


class RocketSimEnv(gym.Env):

    def __init__(self, design):
        super().__init__()
        self.action_space = gym.spaces.Discrete(3)
        low = np.array([-np.inf,    # p0 x
                        -np.inf,    # p1 y
                        -1.0,       # p2 z
                        -np.inf,    # v0 vx
                        -np.inf,    # v1 vy
                        -np.inf,    # v2 vz
                        -np.inf,    # q0
                        -np.inf,    # q1
                        -np.inf,    # q2
                        -np.inf,    # q3
                        -np.inf,    # o0 ox
                        -np.inf,    # o1 oy
                        -np.inf     # o2 0z
                        ])
        high = np.array([np.inf,  # p0 x
                         np.inf,  # p1 y
                         np.inf,  # p2 z
                         np.inf,  # v0 vx
                         np.inf,  # v1 vy
                         np.inf,  # v2 vz
                         np.inf,  # q0
                         np.inf,  # q1
                         np.inf,  # q2
                         np.inf,  # q3
                         np.inf,  # o0 ox
                         np.inf,  # o1 oy
                         np.inf  # o2 0z
                         ])
        self.observation_space = gym.spaces.Box(low=low, high=high)

        # 観測ステップ
        self.obc_step = 10

        # 計算設定
        self.T = 150
        self.dt = 0.01
        self.time = np.arange(0.0, self.T, self.dt)
        self.N = len(self.time)

        # 機体設計
        self.name = design['name']
        self.m_af = design['m_af']
        self.I_af = design['I_af']
        self.CP = design['CP']
        self.CG_a = design['CG_a']
        self.d = design['d']
        self.area = np.pi * (self.d ** 2) / 4.0
        self.len_a = design['len_a']
        self.inertia_z0 = design['inertia_z0']
        self.inertia_zT = design['inertia_zT']
        self.engine = design['engine']
        self.me_total = design['me_total']
        self.me_prop = design['me_prop']
        self.len_e = design['len_e']
        self.d_e = design['d_e']

        # 打ち上げ条件
        self.p0 = np.array([0., 0., 0.])  # position(x, y, z)
        # self.condition_name = launch_condition['name']
        self.theta0 = np.deg2rad(0)
        self.phi0 = 0
        self.launch_rod = 1
        self.v0 = np.array([0., 0., 0.])  # velocity(vx, vy, vz)
        self.ome0 = np.array([0., 0., 0.])
        self.density = 1.26
        self.wind_R = np.random.rand() * 8  # [0, 8]
        self.z_R = 1
        self.beta = np.random.choice([-np.pi/2, np.pi/2])  # wind direction {-np.pi/2, np.pi/2}

        self.qt = Quaternion()
        self.CPy = [0, -self.d/4, self.d/4]

        self.cal_method = 'Euler'

        self.world_scale = 1

        self.viewer = None

    def _reset(self):
        self.wind_direction = np.array([np.cos(self.beta), np.sin(self.beta), 0.0])
        self.qua_theta0 = np.array([np.cos(0.5 * self.theta0), np.sin(0.5 * self.theta0), 0., 0.])  # x軸theta[rad]回転, 射角
        self.qua_phi0 = np.array([np.cos(0.5 * self.phi0), 0., 0., np.sin(0.5 * self.phi0)])  # z軸phi[rad]回転, 方位角
        self.wind_direction = np.array([np.cos(self.beta), np.sin(self.beta), 0.0])

        self.engine_data = np.loadtxt(self.engine)

        self.force = Force(self.area, self.engine_data, self.T, self.density)
        self.thrust = self.force.thrust()

        self.mass = Mass(self.m_af, self.I_af, self.CG_a, self.len_a, self.inertia_z0, self.inertia_zT, self.me_total,
                         self.me_prop, self.len_e, self.d_e, self.force.burn_time, self.T)
        self.M = self.mass.mass()
        self.Me = self.mass.me_t()
        self.Me_dot = self.mass.me_dot()
        self.CG = self.mass.CG()
        self.CG_dot = self.mass.CG_dot()
        self.Ie = self.mass.iexg()
        self.Inertia = self.mass.inertia()
        self.Inertia_z = self.mass.inertia_z()
        self.Inertia_dot = self.mass.inertia_dot()
        self.Inertia_z_dot = self.mass.inertia_z_dot()

        self.wind = Wind(self.z_R, self.wind_R)
        v_air = self.wind.wind(self.p0[2]) * self.wind_direction - self.v0

        self.rocpath = []
        self.n_step = 0
        self.state = np.r_[self.p0, self.v0, self.qt.product(self.qua_phi0, self.qua_theta0), self.ome0]

        return self.state

    def _step(self, action):
        self.state, self.done = self._observe(action, n_step=self.n_step)
        if self.done:
            self.reward = self.state[2]  # -np.abs(np.arctan(self.state[1] / self.state[2])) * 10   # -arctan(y/z)
        else:
            self.reward = 0     # -np.abs(np.arctan(self.state[1] / self.state[2]))
        self.n_step += 1
        return self.state, self.reward, self.done, {}

    def _observe(self, action,  n_step):
        done = False
        pi = self.state[:3]
        vi = self.state[3:6]
        quai = self.state[6:10]
        omei = self.state[10:13]

        self.CPy_action = self.CPy[action]

        for (i, t) in enumerate(np.arange(self.dt * n_step * self.obc_step, self.dt * (n_step + 1) * self.obc_step, self.dt)):
            if self.cal_method == 'Euler':
                # Forward Euler
                p_dot, v_dot, qua_dot, ome_dot = self.deriv(pi, vi, quai, omei, t)
                if np.isnan(qua_dot).any() or np.isinf(qua_dot).any() or np.isnan(ome_dot).any() or np.isinf(ome_dot).any():
                    done = True
                    break
                pi += vi * self.dt
                vi += v_dot * self.dt
                quai += qua_dot * self.dt
                omei += ome_dot * self.dt

            elif self.cal_method == 'RungeKutta':
                p_dot0, v_dot0, qua_dot0, ome_dot0 = self.deriv(pi, vi, quai, omei, t)
                if np.isnan(qua_dot0).any() or np.isinf(qua_dot0).any() or np.isnan(ome_dot0).any() or np.isinf(ome_dot0).any():
                    done = True
                    break
                p_dot1, v_dot1, qua_dot1, ome_dot1 = self.deriv(pi + p_dot0 * self.dt / 2, vi + v_dot0 * self.dt / 2, quai + qua_dot0 * self.dt / 2, omei + ome_dot0 * self.dt / 2, t + self.dt / 2)
                p_dot2, v_dot2, qua_dot2, ome_dot2 = self.deriv(pi + p_dot1 * self.dt / 2, vi + v_dot1 * self.dt / 2, quai + qua_dot1 * self.dt / 2, omei + ome_dot1 * self.dt / 2, t + self.dt / 2)
                p_dot3, v_dot3, qua_dot3, ome_dot3 = self.deriv(pi + p_dot2 * self.dt, vi + v_dot2 * self.dt, quai + qua_dot2 * self.dt, omei + ome_dot2 * self.dt, t + self.dt)
                pi += (p_dot0 + 2 * p_dot1 + 2 * p_dot2 + p_dot3) * self.dt / 6
                vi += (v_dot0 + 2 * v_dot1 + 2 * v_dot2 + v_dot3) * self.dt / 6
                quai += (qua_dot0 + 2 * qua_dot1 + 2 * qua_dot2 + qua_dot3) * self.dt / 6
                omei += (ome_dot0 + 2 * ome_dot1 + 2 * ome_dot2 + ome_dot3) * self.dt / 6

            self.wv = self.wind.wind(pi[2])
            v_air = self.wv * self.wind_direction - vi

            # ランチロッド離脱後，vz<=0となる頂点で計算終了
            if np.linalg.norm(pi) >= self.launch_rod and vi[2] <= 0 and pi[2] <= 0:
                done = True
                break

            quai /= np.linalg.norm(quai)

            if t <= self.force.burn_time:
                pi[2] = max(0., pi[2])

        return np.r_[pi, vi, quai, omei], done

    def deriv(self, pi, vi, quai, omei, t):
        # 機軸座標系の推力方向ベクトル
        r_Ta = np.array([0., 0., 1.0])
        # 慣性座標系重力加速度
        gra = np.array([0., 0., -g])
        # 機軸座標系の空力中心位置
        r = np.array([0., self.CPy_action, self.CG(t) - self.CP])
        # 慣性座標系の推力方向ベクトル
        r_T = self.qt.rotation(r_Ta, self.qt.coquat(quai))
        r_T /= np.linalg.norm(r_T)
        # 慣性テンソル
        I = np.diag([self.Inertia(t), self.Inertia(t), self.Inertia_z(t)])
        # 慣性テンソルの時間微分
        I_dot = np.diag([self.Inertia_dot(t), self.Inertia_dot(t), self.Inertia_z_dot(t)])
        # 慣性座標系対気速度
        v_air = self.wind.wind(pi[2]) * self.wind_direction - vi
        # 迎角
        alpha = aoa.aoa(self.qt.rotation(v_air, quai))
        # ランチロッド垂直抗力
        N = 0
        # ランチロッド進行中
        if np.linalg.norm(pi) <= self.launch_rod and r_T[2] >= 0:
            Mg_ = self.M(t) * gra - np.dot(self.M(t) * gra, r_T) * r_T
            D_ = self.force.drag(alpha, v_air) - np.dot(self.force.drag(alpha, v_air), r_T) * r_T
            N = -Mg_ - D_
        # 慣性座標系加速度
        v_dot = gra + (self.thrust(t) * r_T + self.force.drag(alpha, v_air) + N) / self.M(t)
        # クォータニオンの導関数
        qua_dot = self.qt.qua_dot(omei, quai)
        # 機軸座標系角加速度
        ome_dot = np.linalg.solve(I, - np.cross(r, self.qt.rotation(self.force.drag(alpha, v_air), quai))
                                  - np.dot(I_dot, omei) - np.cross(omei, np.dot(I, omei)))
        # ランチロッド進行中
        if np.linalg.norm(pi) <= self.launch_rod:
            # ランチロッド進行中は姿勢が一定なので角加速度0とする
            ome_dot = np.array([0., 0., 0.])

        return vi, v_dot, qua_dot, ome_dot

    def _render(self, mode='human', close=False):
        screen_width = 700
        screen_height = 700
        t = self.dt * self.n_step * self.obc_step
        CG_rate = self.CG(t) / self.len_a

        if self.viewer is None:
            self.viewer = rendering.Viewer(screen_width, screen_height)
        self.viewer.geoms = []

        self.viewer.draw_polyline([[0, screen_height / 10], [screen_width, screen_height / 10]])

        ground = rendering.FilledPolygon([(0, screen_height / 10), (0, 0), (screen_width, 0), (screen_width, screen_height / 10)])
        ground.set_color(.84, .56, .34)
        self.viewer.add_geom(ground)

        sky = rendering.FilledPolygon(
            [(0, screen_height), (0, screen_height / 10), (screen_width, screen_height / 10), (screen_width, screen_height)])
        sky.set_color(.63, .85, .94)
        self.viewer.add_geom(sky)

        y = self.state[1] * self.world_scale + screen_width / 2
        z = self.state[2] * self.world_scale + screen_height / 10
        self.rocpath.append([y, z])
        self.viewer.draw_polyline(self.rocpath)

        l, w, ln, lf, wf = 200, 40, 50, 20, 20
        posrocket = rendering.FilledPolygon([(0, l / 2 - (0.5 - CG_rate) * l), (-w / 2, l / 2 - ln - (0.5 - CG_rate) * l),
                                          (-w / 2, -l / 2 - (0.5 - CG_rate) * l),
                                          (w / 2, -l / 2 - (0.5 - CG_rate) * l),
                                          (w / 2, (l / 2 - ln - (0.5 - CG_rate) * l))])
        posrocket.set_color(.5, .5, .5)
        self.posrockettrans = rendering.Transform(translation=(y, z), scale=(0.3, 0.3))
        posrocket.add_attr(self.posrockettrans)
        self.viewer.add_geom(posrocket)

        posfin = rendering.FilledPolygon(
            [(-w / 2, -(l / 2 - lf) - (0.5 - CG_rate) * l), (-(w / 2 + wf), -l / 2 - (0.5 - CG_rate) * l),
             (w / 2 + wf, -l / 2 - (0.5 - CG_rate) * l), (w / 2, -(l / 2 - lf) - (0.5 - CG_rate) * l)])
        posfin.set_color(.5, .5, .5)
        posfin.add_attr(self.posrockettrans)
        self.viewer.add_geom(posfin)

        if t <= self.force.burn_time:
            posflame = rendering.FilledPolygon(
                [(-w / 4, -l / 2 - (0.5 - CG_rate) * l), (0, -l / 2 - l / 8 - (0.5 - CG_rate) * l),
                 (w / 4, -l / 2 - (0.5 - CG_rate) * l)])
            posflame.set_color(.9, .0, .0)
            posflame.add_attr(self.posrockettrans)
            self.viewer.add_geom(posflame)

        r_Ta = np.array([0., 0., 1.0])
        quai = self.state[6:10]
        r_T = self.qt.rotation(r_Ta, self.qt.coquat(quai))
        x_roll = 0.
        if not r_T[1] == 0.:
            x_roll = np.arctan(r_T[2] / r_T[1]) - np.pi / 2 * np.sign(r_T[1])
        self.posrockettrans.set_rotation(x_roll)

        return self.viewer.render(return_rgb_array=(mode == 'rgb_array'))

    def _close(self):
            if self.viewer:
                self.viewer.close()


if __name__ == '__main__':
    import pandas as pd

    design = pd.read_csv('design.csv')
    rsEnv = RocketSimEnv(design.loc[0])
    rsEnv.obc_step = 1
    rsEnv.cal_method = 'RungeKutta'
    rsEnv.wind_R = 3.
    theta0 = 3
    rsEnv.theta0 = np.deg2rad(theta0)
    rsEnv.beta = np.pi / 2

    rsEnv.reset()
    action = 0

    for i in range(rsEnv.N // rsEnv.obc_step):
        state, reward, done, _ = rsEnv.step(action)
        rsEnv.render(close=True)
        if done:
            break
