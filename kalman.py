import numpy as np
from environment import Obstacle

class KalmanFilter:
    def __init__(self, env):
        self.env = env
        self.sigma = 0.01*np.eye(3, dtype=float)
        self.R = np.diag([0.0001, 0.0001, 0.001]) # sigma x, y, a
        self.Q_obstacles = np.diag([0.1, np.deg2rad(2), 0.001]) # sigma r, phi, color
        self.Q_odometry = np.diag([0.001, 0.001, 0.001]) # sigma x, y, theta
        
        # acceptable mahalanobis distance
        self.alpha = 2
        
        self.mu = np.array([[self.env.robot.x],[self.env.robot.y],[self.env.robot.a]], dtype=np.float64)

    def Fx(self):
        total_state_length = 3 + 3 * len(self.env.obstacles)  # 3 for robot state, 3 per obstacle
        Fx_base = np.eye(3)  # Base for robot state
        Fx_obstacles = np.zeros((3, total_state_length - 3))  # Extend to cover all obstacles
        return np.block([Fx_base, Fx_obstacles])  # Combined state transition matrix

    def prediction_update(self, u, dt):
        v, w = u[0], u[1]
        theta = self.mu[2]
        if abs(w) > 1e-5:
            dx = -(v / w) * np.sin(theta) + (v / w) * np.sin(theta + w * dt)
            dy = (v / w) * np.cos(theta) - (v / w) * np.cos(theta + w * dt)
            dtheta = w * dt
        else:
            dx = v * dt * np.cos(theta)
            dy = v * dt * np.sin(theta)
            dtheta = 0
            
        self.mu[0] += dx          
        self.mu[1] += dy          
        self.mu[2] += dtheta          
        self.mu[2] = np.arctan2(np.sin(self.mu[2]), np.cos(self.mu[2]))                              
        # Jacobian matrix G for state transition
        G = np.zeros((3, 3))
        if abs(w) > 1e-5:
            G[0, 2] = -(v / w) * np.cos(theta) + (v / w) * np.cos(theta + w * dt)
            G[1, 2] = -(v / w) * np.sin(theta) + (v / w) * np.sin(theta + w * dt)
        else:
            G[0, 2] = -v * dt * np.sin(theta)
            G[1, 2] = v * dt * np.cos(theta)
            
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
        q = np.linalg.norm(delta)**2
        
        sqrt_q = np.sqrt(q)
        angle = np.arctan2(delta[1, 0], delta[0, 0]) - self.env.robot.a
        # angle = np.arctan2(np.sin(angle), np.cos(angle))
        current_color = self.mu[obstacle_idx + 2]
        
        z_hat = np.vstack((sqrt_q, angle, current_color)).reshape(3, 1)
        
        z_actual = np.array([[dist], [phi], [color]], dtype = np.float64)
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
                if mahalanobis_distance < self.alpha and mahalanobis_distance < min_distance:
                    min_distance = mahalanobis_distance
                    closest_idx = idx * 3 + 3
                    
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
                
                self.mu[closest_idx] = measured_x
                self.mu[closest_idx + 1] = measured_y
                self.mu[closest_idx + 2] = color
                
            # Update the state and covariance matrices
            delta = np.array([[self.mu[closest_idx] - self.env.robot.x],
                              [self.mu[closest_idx + 1] - self.env.robot.y]])
            q = np.linalg.norm(delta)**2

            sqrt_q = np.sqrt(q)
            angle = np.arctan2(delta[1, 0], delta[0, 0]) - self.env.robot.a
            # angle = np.arctan2(np.sin(angle), np.cos(angle))
            current_color = self.mu[closest_idx + 2]
            z_hat = np.vstack((sqrt_q, angle, current_color)).reshape(3, 1)
            
            
            z_actual = np.array([[dist], [phi], [color]], dtype = np.float64)
            z_actual = np.reshape(z_actual, (3, 1))
            
            z_delta = z_actual - z_hat
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

    def odometry_measurement_update(self, z_odometry):
        # Measurement matrix
        H = np.zeros((3, self.mu.shape[0]))
        H[:3, :3] = np.eye(3)  # assuming the first three state variables are x, y, theta

        # Measurement residual
        dx = z_odometry[0] - self.mu[0]
        dy = z_odometry[1] - self.mu[1]
        dtheta = z_odometry[2] - self.mu[2]
        y = np.array([[dx], [dy], [dtheta]], dtype = np.float64)
        y = np.reshape(y, (3, 1))
        
        # Normalization of the angle difference
        y[2] = np.arctan2(np.sin(y[2]), np.cos(y[2]))

        # Compute the Kalman gain
        S = H @ self.sigma @ H.T + self.Q_odometry
        K = self.sigma @ H.T @ np.linalg.inv(S)
        # Update the state estimate.
        self.mu += K @ y
        self.mu[2] = np.arctan2(np.sin(self.mu[2]), np.cos(self.mu[2]))  # Normalize the angle

        # Update the covariance matrix.
        self.sigma = (np.eye(len(self.sigma)) - K @ H) @ self.sigma

        return


    def update_environment(self):
        self.env.robot.x = self.mu[0]
        self.env.robot.y = self.mu[1]
        self.env.robot.a = self.mu[2]
        for idx, obstacle in enumerate(self.env.obstacles):
            mu_idx = 3 + idx * 3
            obstacle.x = self.mu[mu_idx]
            obstacle.y = self.mu[mu_idx + 1]
            obstacle.color = self.mu[mu_idx + 2]
            
    def process_measurement(self, u, z_obstacles, z_odometry, dt):
        self.prediction_update(u, dt)
        self.obstacles_measurement_update(z_obstacles)
        self.odometry_measurement_update(z_odometry)
        self.update_environment()