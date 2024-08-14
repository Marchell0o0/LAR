import numpy as np
from environment import Obstacle


class KalmanFilter:
    def __init__(self, env):
        self.env = env
        self.sigma = 0.001 * np.eye(3, dtype=float)
        # self.sigma = 0.01 * np.eye(3, dtype=float)
        self.mu = np.array([[self.env.robot.x], [self.env.robot.y], [self.env.robot.a]], dtype=np.float64)

        self.R = np.diag([0.00001, 0.00001, 0.0001])  # sigma x, y, a
        self.Q_obstacles = np.diag([0.05, np.deg2rad(1), 0.00001])  # sigma r, phi, color

        # acceptable mahalanobis distance
        self.alpha = 4
        self.alpha_for_green = 0.4

    def Fx(self):
        total_state_length = 3 + 3 * len(self.env.obstacles)  # 3 for robot state, 3 per obstacle
        Fx_base = np.eye(3)
        Fx_obstacles = np.zeros((3, total_state_length - 3))
        return np.block([Fx_base, Fx_obstacles])

    def pos_update(self, u):
        dx = u[0]
        dy = u[1]
        dtheta = u[2]
        theta = self.mu[2]  # heading before update

        self.mu[0] += dx
        self.mu[1] += dy
        self.mu[2] += dtheta
        self.mu[2] = np.arctan2(np.sin(self.mu[2]), np.cos(self.mu[2]))

        # Jacobian matrix G for state transition
        G = np.zeros((3, 3))

        # Alternate taylor expansion
        # G[0, 2] = -distance_travelled * np.sin(theta + dtheta / 2)
        # G[1, 2] = distance_travelled * np.cos(theta + dtheta / 2)
        G[0, 2] = -dx * np.sin(theta) - dy * np.cos(theta)
        G[1, 2] = dx * np.cos(theta) - dy * np.sin(theta)

        Fx = self.Fx()
        G = np.eye(self.sigma.shape[0]) + Fx.T @ G @ Fx
        self.sigma = G @ self.sigma @ G.T
        self.sigma = self.sigma + Fx.T @ self.R @ Fx

        return

    def compute_mahalanobis_distance(self, measurement, obstacle_idx):
        dist, phi, color = measurement
        if isinstance(dist, np.ndarray):
            dist = dist.item()
        if isinstance(phi, np.ndarray):
            phi = phi.item()
        if isinstance(color, np.ndarray):
            color = color.item()

        delta = np.array([[self.mu[obstacle_idx] - self.env.robot.x],
                          [self.mu[obstacle_idx + 1] - self.env.robot.y]])
        q = np.linalg.norm(delta) ** 2

        sqrt_q = np.sqrt(q)
        angle = np.arctan2(delta[1, 0], delta[0, 0]) - self.env.robot.a
        current_color = self.mu[obstacle_idx + 2]

        z_hat = np.vstack((sqrt_q, angle, current_color)).reshape(3, 1)

        z_actual = np.array([[dist], [phi], [color]], dtype=np.float64)
        z_actual = np.reshape(z_actual, (3, 1))

        z_delta = (z_actual - z_hat)
        z_delta[1] = np.arctan2(np.sin(z_delta[1]), np.cos(z_delta[1]))
        H = self.calculate_jacobian(delta, q, obstacle_idx)
        psi = H @ self.sigma @ H.T + self.Q_obstacles
        pi = z_delta.T @ np.linalg.inv(psi) @ z_delta
        return pi

    def obstacles_measurement_update(self, zs):
        for z in zs:
            dist, phi, color = z

            measured_x = self.mu[0] + dist * np.cos(phi + self.mu[2])
            measured_y = self.mu[1] + dist * np.sin(phi + self.mu[2])
            measured_obstacle = Obstacle(measured_x, measured_y, color)
            closest_idx = -1
            min_distance = float('inf')
            for idx, obstacle in enumerate(self.env.obstacles):
                mahalanobis_distance = self.compute_mahalanobis_distance(z, idx * 3 + 3)
                if color == 2 and obstacle.color == 2:
                    if mahalanobis_distance < self.alpha_for_green and mahalanobis_distance < min_distance:
                        min_distance = mahalanobis_distance
                        closest_idx = idx * 3 + 3
                        current_obstacle = obstacle
                else:
                    if mahalanobis_distance < self.alpha and mahalanobis_distance < min_distance:
                        min_distance = mahalanobis_distance
                        closest_idx = idx * 3 + 3
                        current_obstacle = obstacle

            if closest_idx == -1:
                print("Adding new obstacle")
                print(measured_obstacle)
                closest_idx = self.mu.shape[0]
                new_size = closest_idx + 3

                # Expand the state vector
                extension = np.zeros((3, 1))
                self.mu = np.vstack((self.mu, extension))

                # Expand the covariance matrix
                sigma_extension = np.zeros((new_size, new_size))
                sigma_extension[:self.sigma.shape[0], :self.sigma.shape[1]] = self.sigma
                new_diag = 100 * np.eye(3)
                sigma_extension[-3:, -3:] = new_diag
                self.sigma = sigma_extension

                # Initialize the new obstacle
                self.env.obstacles.append(measured_obstacle)
                self.env.obstacles_measurement_count[measured_obstacle] = 1

                self.mu[closest_idx] = measured_x
                self.mu[closest_idx + 1] = measured_y
                self.mu[closest_idx + 2] = color
            else:
                self.env.obstacles_measurement_count[current_obstacle] += 1

            # Update the state and covariance matrices
            delta = np.array([[self.mu[closest_idx] - self.env.robot.x],
                              [self.mu[closest_idx + 1] - self.env.robot.y]])
            q = np.linalg.norm(delta) ** 2

            sqrt_q = np.sqrt(q)
            angle = np.arctan2(delta[1, 0], delta[0, 0]) - self.env.robot.a

            current_color = self.mu[closest_idx + 2]
            z_hat = np.vstack((sqrt_q, angle, current_color)).reshape(3, 1)

            z_actual = np.array([[dist], [phi], [color]], dtype=np.float64)
            z_actual = np.reshape(z_actual, (3, 1))

            z_delta = z_actual - z_hat
            # print("Difference of measurement from current data for:", measured_obstacle)
            z_delta[1] = np.arctan2(np.sin(z_delta[1]), np.cos(z_delta[1]))

            H = self.calculate_jacobian(delta, q, closest_idx)
            S = H @ self.sigma @ H.T + self.Q_obstacles
            K = self.sigma @ H.T @ np.linalg.inv(S)

            self.mu += K @ z_delta
            self.mu[2] = np.arctan2(np.sin(self.mu[2]), np.cos(self.mu[2]))
            self.sigma = (np.eye(self.sigma.shape[0]) - K @ H) @ self.sigma

    def calculate_jacobian(self, delta, q, idx):
        sqrt_q = np.sqrt(q)
        H = np.zeros((3, self.sigma.shape[0]))
        H[0, 0] = -delta[0] / sqrt_q
        H[0, 1] = -delta[1] / sqrt_q
        H[1, 0] = delta[1] / q
        H[1, 1] = -delta[0] / q
        H[1, 2] = -1

        H[0, idx] = delta[0] / sqrt_q
        H[0, idx + 1] = delta[1] / sqrt_q
        H[1, idx] = -delta[1] / q
        H[1, idx + 1] = delta[0] / q
        H[2, idx + 2] = 1
        return H

    def update_for_visualization(self):
        self.env.robot.x = self.mu[0]
        self.env.robot.y = self.mu[1]
        self.env.robot.a = self.mu[2]
        for idx, obstacle in enumerate(self.env.obstacles):
            mu_idx = 3 + idx * 3
            obstacle.x = self.mu[mu_idx]
            obstacle.y = self.mu[mu_idx + 1]
            obstacle.color = self.mu[mu_idx + 2]