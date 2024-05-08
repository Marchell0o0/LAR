import numpy as np
from environment import Obstacle

class KalmanFilter:
    def __init__(self, env):
        self.env = env
        self.sigma = 0.01*np.eye(3, dtype=float)
        self.R = np.diag([0.001, 0.001, 0.001]) # sigma x, y, a
        self.Q = np.diag([0.1, np.deg2rad(20)]) # sigma r, phi, color
        
        # acceptable mahalanobis distance
        self.alpha = 1
        
        self.mu = np.array([[self.env.robot.x],[self.env.robot.y],[self.env.robot.a]], dtype=np.float64)

    def Fx(self):
        total_state_length = 3 + 2 * len(self.env.obstacles)  # 3 for robot state, 3 per obstacle
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
        
        # min_q = 1e-6
        # q = max(q, min_q)
        
        sqrt_q = np.sqrt(q)
        angle = np.arctan2(delta[1, 0], delta[0, 0]) - self.env.robot.a
        angle = np.arctan2(np.sin(angle), np.cos(angle))
        
        z_hat = np.vstack((sqrt_q, angle)).reshape(2, 1)
        
        z_actual = np.array([[dist], [phi]], dtype = np.float64)
        z_actual = np.reshape(z_actual, (2, 1))
        
        z_delta = (z_actual - z_hat)
        H = self.calculate_jacobian(delta, q, obstacle_idx)
        # print(f"delta: {delta}")
        # print(f"q: {q}")
        # print(f"H: {H}")
        # z_delta[2] = 0
        psi = H @ self.sigma @ H.T + self.Q
        # print(psi)
        # distance for comparison to assign 
        pi = z_delta.T @ np.linalg.inv(psi) @ z_delta
        return pi

    def measurement_update(self, zs):
        for z in zs:
            dist, phi, color = z
            
            measured_x = self.mu[0] + dist * np.cos(phi + self.mu[2])
            measured_y = self.mu[1] + dist * np.sin(phi + self.mu[2])
            measured_obstacle = Obstacle(measured_x, measured_y, color)
            print("Measured ", measured_obstacle)
            # Find the closest obstacle based on Mahalanobis distance
            closest_idx = -1
            min_distance = float('inf')
            for idx, obstacle in enumerate(self.env.obstacles):
                mahalanobis_distance = self.compute_mahalanobis_distance(z, idx * 2 + 3)
                if mahalanobis_distance < self.alpha and mahalanobis_distance < min_distance:
                    min_distance = mahalanobis_distance
                    closest_idx = idx * 2 + 3
                    
            added_new = False
            # Expand the state vector and covariance matrix if necessary
            if closest_idx == -1:
                added_new = True
                distance_to_robot = np.sqrt((measured_obstacle.x - self.env.robot.x)**2 + (measured_obstacle.y - self.env.robot.y)**2)
                closest_idx = self.mu.shape[0]
                new_size = closest_idx + 2
                
                # Expand the state vector
                extension = np.zeros((2, 1))
                self.mu = np.vstack((self.mu, extension))
                
                # Expand the covariance matrix
                sigma_extension = np.zeros((new_size, new_size))
                sigma_extension[:self.sigma.shape[0], :self.sigma.shape[1]] = self.sigma
                new_diag = 100 * np.eye(2)
                sigma_extension[-2:, -2:] = new_diag
                self.sigma = sigma_extension
                
                # Initialize the new obstacle
                self.env.obstacles.append(measured_obstacle)
                
                self.mu[closest_idx] = measured_x
                self.mu[closest_idx + 1] = measured_y
                    # self.mu[closest_idx + 2] = color
                
            # Update the state and covariance matrices
            delta = np.array([[self.mu[closest_idx] - self.env.robot.x],
                              [self.mu[closest_idx + 1] - self.env.robot.y]])
            q = np.linalg.norm(delta)**2

            sqrt_q = np.sqrt(q)
            angle = np.arctan2(delta[1, 0], delta[0, 0]) - self.env.robot.a
            angle = np.arctan2(np.sin(angle), np.cos(angle))
            # current_color = self.mu[closest_idx + 2]
            
            z_hat = np.vstack((sqrt_q, angle)).reshape(2, 1)
            
            z_actual = np.array([[dist], [phi]], dtype = np.float64)
            z_actual = np.reshape(z_actual, (2, 1))
            
            z_delta = z_actual - z_hat
            H = self.calculate_jacobian(delta, q, closest_idx)
            
            S = H @ self.sigma @ H.T + self.Q
            K = self.sigma @ H.T @ np.linalg.inv(S)
            

            self.mu += K @ z_delta
            self.mu[2] = np.arctan2(np.sin(self.mu[2]), np.cos(self.mu[2]))                              
            self.sigma = (np.eye(self.sigma.shape[0]) - K @ H) @ self.sigma
                
    def calculate_jacobian(self, delta, q, idx):
        sqrt_q = np.sqrt(q)
        # Compute the Jacobian matrix for the measurement function
        H = np.zeros((2, self.sigma.shape[0]))
        H[0, 0] = -delta[0] / sqrt_q
        H[0, 1] = -delta[1] / sqrt_q
        H[1, 0] = delta[1] / q
        H[1, 1] = -delta[0] / q
        H[1, 2] = -1
        
        H[0, idx] = delta[0] / sqrt_q
        H[0, idx + 1] = delta[1] / sqrt_q
        H[1, idx] = -delta[1] / q
        H[1, idx + 1] = delta[0] / q
        # H[2, idx + 2] = 1
        return H

    def update_environment(self):
        self.env.robot.x = self.mu[0]
        self.env.robot.y = self.mu[1]
        self.env.robot.a = self.mu[2]
        for idx, obstacle in enumerate(self.env.obstacles):
            mu_idx = 3 + idx * 2
            obstacle.x = self.mu[mu_idx]
            obstacle.y = self.mu[mu_idx + 1]
            # obstacle.color = self.mu[mu_idx + 2]
            
    def process_measurement(self, u, z, dt):
        self.prediction_update(u, dt)
        self.measurement_update(z)
        self.update_environment()


# import numpy as np
# from environment import Obstacle

# class KalmanFilter:
#     def __init__(self, env):
#         self.env = env
#         self.sigma = 0.01*np.eye(3, dtype=float)
#         self.R = np.diag([0.0001, 0.0001, 0.0001]) # sigma x, y, a
#         self.Q = np.diag([0.1, 0.2, 0.01]) # sigma r, phi, color
        
#         #acceptable mahalanobis distance
#         self.alpha = 7.8
        
#         self.mu = np.array([[self.env.robot.x],[self.env.robot.y],[self.env.robot.a]], dtype=np.float64)

#     def Fx(self):
#         total_state_length = 3 + 3 * len(self.env.obstacles)  # 3 for robot state, 3 per obstacle
#         Fx_base = np.eye(3)  # Base for robot state
#         Fx_obstacles = np.zeros((3, total_state_length - 3))  # Extend to cover all obstacles
#         return np.block([Fx_base, Fx_obstacles])  # Combined state transition matrix

#     def prediction_update(self, u, dt):
#         v, w = u[0], u[1]
#         theta = self.mu[2]
#         if abs(w) > 1e-5:
#             dx = -(v / w) * np.sin(theta) + (v / w) * np.sin(theta + w * dt)
#             dy = (v / w) * np.cos(theta) - (v / w) * np.cos(theta + w * dt)
#             dtheta = w * dt
#         else:
#             dx = v * dt * np.cos(theta)
#             dy = v * dt * np.sin(theta)
#             dtheta = 0
            
#         self.mu[0] += dx          
#         self.mu[1] += dy          
#         self.mu[2] += dtheta          
#         self.mu[2] = np.arctan2(np.sin(self.mu[2]), np.cos(self.mu[2]))                              
#         # Jacobian matrix G for state transition
#         G = np.zeros((3, 3))
#         if abs(w) > 1e-5:
#             G[0, 2] = -(v / w) * np.cos(theta) + (v / w) * np.cos(theta + w * dt)
#             G[1, 2] = -(v / w) * np.sin(theta) + (v / w) * np.sin(theta + w * dt)
#         else:
#             G[0, 2] = -v * dt * np.sin(theta)
#             G[1, 2] = v * dt * np.cos(theta)
            
#         Fx = self.Fx()
#         G = np.eye(self.sigma.shape[0]) + Fx.T @ G @ Fx
#         self.sigma = G @ self.sigma @ G.T
#         self.sigma = self.sigma + Fx.T @ self.R @ Fx
        
#         return

#     def compute_mahalanobis_distance(self, measurement, obstacle_idx, robot):
#         dist, phi, color = measurement
#         if isinstance(dist, np.ndarray):
#                 dist = dist.item()
#         if isinstance(phi, np.ndarray):
#             phi = phi.item()
#         if isinstance(color, np.ndarray):
#             color = color.item()
            
#         delta = np.array([[self.mu[obstacle_idx] - self.env.robot.x], [self.mu[obstacle_idx + 1] - self.env.robot.y]])
#         # print(self.mu[obstacle_idx], self.mu[obstacle_idx + 1])
#         q = np.linalg.norm(delta)**2
        
#         min_q = 1e-6
#         q = max(q, min_q)
        
#         sqrt_q = np.sqrt(q)
#         angle = np.arctan2(delta[1, 0], delta[0, 0]) - self.env.robot.a
#         current_color = self.mu[obstacle_idx + 2]
#         z_hat = np.vstack((sqrt_q, angle, current_color)).reshape(3, 1)
        
#         z_actual = np.array([[dist], [phi], [color]], dtype = np.float64)
#         z_actual = np.reshape(z_actual, (3, 1))
        
#         z_delta = (z_actual - z_hat)
#         H = self.calculate_jacobian(delta, q, obstacle_idx)
#         # print(f"delta: {delta}")
#         # print(f"q: {q}")
#         # print(f"H: {H}")
#         # z_delta[2] = 0
#         psi = H @ self.sigma @ H.T + self.Q
#         # print(psi)
#         # distance for comparison to assign 
#         pi = z_delta.T @ np.linalg.inv(psi) @ z_delta
#         return pi

#     def measurement_update(self, zs):
#         for z in zs:
#             dist, phi, color = z
#             # if isinstance(dist, np.ndarray):
#             #     dist = dist.item()
#             # if isinstance(phi, np.ndarray):
#             #     phi = phi.item()
#             # if isinstance(color, np.ndarray):
#             #     color = color.item()
            
#             measured_x = self.mu[0] + dist * np.cos(phi + self.mu[2])
#             measured_y = self.mu[1] + dist * np.sin(phi + self.mu[2])
#             measured_obstacle = Obstacle(measured_x, measured_y, color)
#             # print("Measured:")
#             # print(measured_obstacle)
#             # Find the closest obstacle based on Mahalanobis distance
#             closest_idx = -1
#             min_distance = float('inf')
#             # print("Comparing mahalanobis distances")
#             # print("For this obstacle", measured_obstacle)
#             for idx, obstacle in enumerate(self.env.obstacles):
#                 # print(obstacle)
#                 mahalanobis_distance = self.compute_mahalanobis_distance(z, idx * 3 + 3, self.env.robot)
#                 # print("Distance from this obstacle:", mahalanobis_distance)
#                 if mahalanobis_distance < self.alpha and mahalanobis_distance < min_distance:
#                     min_distance = mahalanobis_distance
#                     closest_idx = idx * 3 + 3
#                 # print()
                    
#             # Expand the state vector and covariance matrix if necessary
#             if closest_idx == -1:
#                 if min_distance > 100:
#                     print("That's a big distance")
#                 print("---------------Adding a new obstacle--------------")
#                 print(measured_obstacle)
#                 # No existing obstacle was within the threshold, create a new one
#                 closest_idx = self.mu.shape[0]
#                 new_size = closest_idx + 3
                
#                 # Expand the state vector
#                 extension = np.zeros((3, 1))
#                 self.mu = np.vstack((self.mu, extension))
                
#                 # Expand the covariance matrix
#                 sigma_extension = np.zeros((new_size, new_size))
#                 sigma_extension[:self.sigma.shape[0], :self.sigma.shape[1]] = self.sigma
#                 new_diag = 1 * np.eye(3)
#                 sigma_extension[-3:, -3:] = new_diag
#                 self.sigma = sigma_extension
                
#                 # Initialize the new obstacle
#                 self.env.obstacles.append(measured_obstacle)
                
#                 self.mu[closest_idx] = measured_x
#                 self.mu[closest_idx + 1] = measured_y
#                 self.mu[closest_idx + 2] = color
#                 print(measured_obstacle)
#                 print("list of current obstacles")
#                 for obstacle in self.env.obstacles:
#                     print(obstacle)
                
#             # print("Obstacle  idx:", closest_idx)
#             # Update the state and covariance matrices
#             delta = np.array([[self.mu[closest_idx] - self.env.robot.x], [self.mu[closest_idx + 1] - self.env.robot.y]])
#             q = np.linalg.norm(delta)**2
#             min_q = 1e-6
#             q = max(q, min_q)
#             sqrt_q = np.sqrt(q)
#             angle = np.arctan2(delta[1, 0], delta[0, 0]) - self.env.robot.a
#             current_color = self.mu[closest_idx + 2]
            
#             z_hat = np.vstack((sqrt_q, angle, current_color)).reshape(3, 1)
            
#             z_actual = np.array([[dist], [phi], [color]], dtype = np.float64)
#             z_actual = np.reshape(z_actual, (3, 1))
            
#             z_delta = z_actual - z_hat
#             # print("Z delta: ", z_delta)
#             H = self.calculate_jacobian(delta, q, closest_idx)
#             # print(f"delta: {delta}")
#             # print(f"q: {q}")
#             # print(f"H: {H}")
#             S = H @ self.sigma @ H.T + self.Q
#             K = self.sigma @ H.T @ np.linalg.inv(S)
            
#             # print(f"K: {K}")

#             self.mu += K @ z_delta
#             self.mu[2] = np.arctan2(np.sin(self.mu[2]), np.cos(self.mu[2]))                              
#             self.sigma = (np.eye(self.sigma.shape[0]) - K @ H) @ self.sigma
#             # print(self.mu)
                
#     def calculate_jacobian(self, delta, q, idx):
#         sqrt_q = np.sqrt(q)
#         # Compute the Jacobian matrix for the measurement function
#         H = np.zeros((3, self.sigma.shape[0]))
#         H[0, 0] = -delta[0] / sqrt_q
#         H[0, 1] = -delta[1] / sqrt_q
#         H[1, 0] = delta[1] / q
#         H[1, 1] = -delta[0] / q
#         H[1, 2] = -1
        
#         H[0, idx] = delta[0] / sqrt_q
#         H[0, idx + 1] = delta[1] / sqrt_q
#         H[1, idx] = -delta[1] / q
#         H[1, idx + 1] = delta[0] / q
#         H[2, idx + 2] = 1
#         return H

#     def update_environment(self):
#         self.env.robot.x = self.mu[0]
#         self.env.robot.y = self.mu[1]
#         self.env.robot.a = self.mu[2]
#         for idx, obstacle in enumerate(self.env.obstacles):
#             mu_idx = 3 + idx * 3
#             obstacle.x = self.mu[mu_idx]
#             obstacle.y = self.mu[mu_idx + 1]
#             obstacle.color = self.mu[mu_idx + 2]
            
#     def process_measurement(self, u, z, dt):
#         # print("Processing move:", u)
#         self.prediction_update(u, dt)
        
#         # print("Robot state post-prediction:", self.env.robot.x, self.env.robot.y, self.env.robot.a)
#         # if z:
#             # print("Processing measurement:", z)
#         self.measurement_update(z)
#         self.update_environment()
#         # print("Robot state post-correction:", self.env.robot.x, self.env.robot.y, self.env.robot.a)

