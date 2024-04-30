# import numpy as np
# from environment import Obstacle



    # def measurement_update(self, zs):
    #     '''
    #     This function performs the measurement step of the EKF. Using the linearized observation model, it
    #     updates both the state estimate mu and the state uncertainty sigma based on range and bearing measurements
    #     that are made between robot and landmarks.
    #     Inputs:
    #     - mu: state estimate (robot pose and landmark positions)
    #     - sigma: state uncertainty (covariance matrix)
    #     - zs: list of 3-tuples, (dist,phi,lidx) from measurement function
    #     Outpus:
    #     - mu: updated state estimate
    #     - sigma: updated state uncertainty
    #     '''
    #     delta_zs = [np.zeros((2,1)) for lidx in range(self.N)] # A list of how far an actual measurement is from the estimate measurement
    #     Ks = [np.zeros((mu.shape[0],2)) for lidx in range(self.N)] # A list of matrices stored for use outside the measurement for loop
    #     Hs = [np.zeros((2,mu.shape[0])) for lidx in range(self.N)] # A list of matrices stored for use outside the measurement for loop
    #     for z in zs:
    #         measured = False
    #         x_obstacle = 0
    #         y_obstacle = 0
    #         for obstacle in self.env.obstacles:
    #             # obstacle was measured previously                
    #             if z[3] == obstacle.index:
    #                 measured = True
    #                 x_obstacle = obstacle.x
    #                 y_obstacle = obstacle.y
            
    #         # believe the current measurement
    #         if not measured:
    #             x_obstacle = self.env.robot.x + z[0] * np.cos(z[1] + self.env.robot.a)
    #             y_obstacle = self.env.robot.y + z[0] * np.sin(z[1] + self.env.robot.a)
    #             self.env.obstacles.add(Obstacle(x_obstacle, y_obstacle, z[2], z[3]))
            
    #         # obstacle is surely in the env now
    #         delta  = np.array([[x_obstacle],[y_obstacle]]) - np.array([[self.env.robot.x],[self.env.robot.y]])
    #         q = np.linalg.norm(delta)**2
            
    #         # Distance between robot estimate and and landmark estimate, i.e., distance estimate
    #         dist_est = np.sqrt(q) 
            
    #         # Estimated angled between robot heading and landmark
    #         phi_est = np.arctan2(delta[1,0],delta[0,0]) - self.env.robot.a
    #         phi_est = np.arctan2(np.sin(phi_est),np.cos(phi_est)) 
            
    #         # should add the signature estimation here
    #         z_est_arr = np.array([[dist_est],[phi_est]]) # Estimated observation, in numpy array
    #         z_act_arr = np.array([[z[0]],[z[1]]]) # Actual observation in numpy array
    #         delta_zs[lidx] = z_act_arr - z_est_arr # Difference between actual and estimated observation

    #         # Helper matrices in computing the measurement update
    #         Fxj = np.block([[Fx],[np.zeros((2,Fx.shape[1]))]])
    #         Fxj[3 : 3 + 2,3 + 2*lidx:3 + 2*lidx+2] = np.eye(2)
    #         H = np.array([[-delta[0,0]/np.sqrt(q),-delta[1,0]/np.sqrt(q),0,delta[0,0]/np.sqrt(q),delta[1,0]/np.sqrt(q)],\
    #                     [delta[1,0]/q,-delta[0,0]/q,-1,-delta[1,0]/q,+delta[0,0]/q]])
    #         H = H.dot(Fxj)
    #         Hs[lidx] = H # Added to list of matrices
    #         Ks[lidx] = sigma.dot(np.transpose(H)).dot(np.linalg.inv(H.dot(sigma).dot(np.transpose(H)) + Q)) # Add to list of matrices
        
    #     # After storing appropriate matrices, perform measurement update of mu and sigma
    #     mu_offset = np.zeros(mu.shape) # Offset to be added to state estimate
    #     sigma_factor = np.eye(sigma.shape[0]) # Factor to multiply state uncertainty
    #     for lidx in range(self.N):
    #         mu_offset += Ks[lidx].dot(delta_zs[lidx]) # Compute full mu offset
    #         sigma_factor -= Ks[lidx].dot(Hs[lidx]) # Compute full sigma factor
    #     mu = mu + mu_offset # Update state estimate
    #     sigma = sigma_factor.dot(sigma) # Update state uncertainty
    #     return mu,sigma
