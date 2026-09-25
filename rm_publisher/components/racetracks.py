def generate_racetrack(self):
    self.xy_vels_grid[0,:,:,1] = 0.0
    self.xy_vels_grid[0,:,:,0] = 0.0

    # Straight
    self.xy_vels_grid[0, 1:4, 2:21, 0] = -self.max_velocity
    self.xy_vels_grid[0, 2:7, 1:4, 1] = self.max_velocity
    self.xy_vels_grid[0, 2:11, 19:22, 1] = -self.max_velocity
    self.xy_vels_grid[0, 13:23, 21:24, 1] = -self.max_velocity
    self.xy_vels_grid[0, 21:24, 2:23, 0] = self.max_velocity
    self.xy_vels_grid[0, 10:23, 1:4, 1] = self.max_velocity
    self.xy_vels_grid[0, 9:12, 2:8, 0] = -self.max_velocity
    self.xy_vels_grid[0, 10:19, 6:9, 1] = -self.max_velocity
    self.xy_vels_grid[0, 17:20, 7:19, 0] = -self.max_velocity
    self.xy_vels_grid[0, 14:19, 17:20, 1] = self.max_velocity
    self.xy_vels_grid[0, 13:16, 11:19, 0] = self.max_velocity
    self.xy_vels_grid[0, 6:15, 10:13, 1] = self.max_velocity
    self.xy_vels_grid[0, 5:8, 2:12, 0] = self.max_velocity


    # # Curve 
    self.xy_vels_grid[0, 1:3, 2:5, 1] = self.max_velocity
    self.xy_vels_grid[0, 1, 5:19, 1] = self.max_velocity

    self.xy_vels_grid[0, 4:7, 1:3, 0] = self.max_velocity
    self.xy_vels_grid[0, 3:5, 1, 0] = self.max_velocity

    self.xy_vels_grid[0, 5:7, 9:12, 1] = self.max_velocity
    self.xy_vels_grid[0, 5, 6:10, 1] = self.max_velocity

    self.xy_vels_grid[0, 12:15, 10:12, 0] = self.max_velocity
    self.xy_vels_grid[0, 10:13, 10, 0] = self.max_velocity

    self.xy_vels_grid[0, 13:15, 16:19, 1] = self.max_velocity
    self.xy_vels_grid[0, 13, 15:17, 1] = self.max_velocity

    self.xy_vels_grid[0, 16:19, 18:20, 0] = -self.max_velocity
    self.xy_vels_grid[0, 15:17, 19, 0] = -self.max_velocity

    self.xy_vels_grid[0, 18:20, 7:10, 1] = -self.max_velocity
    self.xy_vels_grid[0, 19, 9:17, 1] = -self.max_velocity

    self.xy_vels_grid[0, 10:13, 7:9, 0] = -self.max_velocity
    self.xy_vels_grid[0, 10:15, 8, 0] = -self.max_velocity

    self.xy_vels_grid[0, 9:11, 2:5, 1] = self.max_velocity
    self.xy_vels_grid[0, 9, 4:6, 1] = self.max_velocity

    self.xy_vels_grid[0, 20:23, 1:3, 0] = self.max_velocity
    self.xy_vels_grid[0, 12:21, 1, 0] = self.max_velocity

    self.xy_vels_grid[0, 22:24, 20:23, 1] = -self.max_velocity
    self.xy_vels_grid[0, 23, 4:21, 1] = -self.max_velocity

    self.xy_vels_grid[0, 13:21, 23, 0] = -self.max_velocity 
    self.xy_vels_grid[0, 12, 21:24, :] = -self.max_velocity
    self.xy_vels_grid[0, 11, 20:23, :] = -self.max_velocity
    self.xy_vels_grid[0, 12, 21, 0] = 0
    self.xy_vels_grid[0, 11, 20, 0] = 0
    self.xy_vels_grid[0, 12, 23, 1] = 0
    self.xy_vels_grid[0, 11, 22, 1] = 0

    self.xy_vels_grid[0, 2:5, 20:22, 0] = -self.max_velocity
    self.xy_vels_grid[0, 4:9, 21, 0] = -self.max_velocity

def generate_racetrack(self):
    self.xy_vels_grid[0,:,:,1] = 0.0
    self.xy_vels_grid[0,:,:,0] = 0.0

    # Middle
    self.xy_vels_grid[0, 2, 3:22, 0] = -self.max_velocity
    self.xy_vels_grid[0, 1:6, 2, 1] = self.max_velocity
    self.xy_vels_grid[0, 6, 1:12, 0] = self.max_velocity
    self.xy_vels_grid[0, 5:14, 12, 1] = self.max_velocity
    self.xy_vels_grid[0, 14, 11:19, 0] = self.max_velocity
    self.xy_vels_grid[0, 13:19, 19, 1] = self.max_velocity
    self.xy_vels_grid[0, 19, 8:21, 0] = -self.max_velocity
    self.xy_vels_grid[0, 11:21, 7, 1] = -self.max_velocity
    self.xy_vels_grid[0, 10, 3:9, 0] = -self.max_velocity
    self.xy_vels_grid[0, 9:22, 2, 1] = self.max_velocity
    self.xy_vels_grid[0, 22, 1:22, 0] = self.max_velocity
    self.xy_vels_grid[0, 13:24, 22, 1] = -self.max_velocity
    self.xy_vels_grid[0, 12, 22, :] = -self.max_velocity
    self.xy_vels_grid[0, 11, 21, :] = -self.max_velocity
    self.xy_vels_grid[0, 3:12, 20, 1] = -self.max_velocity
    
    # Left
    self.xy_vels_grid[0, 1, 3:22, 0] = -self.max_velocity
    self.xy_vels_grid[0, 1, 3:22, 1] = self.max_velocity
    self.xy_vels_grid[0, 1:6, 1, :] = self.max_velocity
    self.xy_vels_grid[0, 7, 1:11, 0] = self.max_velocity
    self.xy_vels_grid[0, 7, 1:11, 1] = -self.max_velocity
    self.xy_vels_grid[0, 7:14, 11, :] = self.max_velocity
    self.xy_vels_grid[0, 15, 11:18, 0] = self.max_velocity
    self.xy_vels_grid[0, 15, 11:18, 1] = -self.max_velocity
    self.xy_vels_grid[0, 15:18, 18, :] = self.max_velocity
    self.xy_vels_grid[0, 18, 9:19, 0] = -self.max_velocity
    self.xy_vels_grid[0, 18, 9:19, 1] = self.max_velocity
    self.xy_vels_grid[0, 11:19, 8, :] = -self.max_velocity
    self.xy_vels_grid[0, 9, 3:9, 0] = -self.max_velocity
    self.xy_vels_grid[0, 9, 3:9, 1] = self.max_velocity
    self.xy_vels_grid[0, 9:22, 1, :] = self.max_velocity
    self.xy_vels_grid[0, 23, 1:22, 0] = self.max_velocity
    self.xy_vels_grid[0, 23, 1:22, 1] = -self.max_velocity
    self.xy_vels_grid[0, 13:24, 23, :] = -self.max_velocity
    self.xy_vels_grid[0, 12, 23, 0] = -self.max_velocity
    self.xy_vels_grid[0, 11, 22, 0] = -self.max_velocity
    self.xy_vels_grid[0, 3:12, 21, :] = -self.max_velocity

    # Right
    self.xy_vels_grid[0, 3, 3:20, :] = -self.max_velocity
    self.xy_vels_grid[0, 3:5, 3, 1] = self.max_velocity
    self.xy_vels_grid[0, 3:5, 3, 0] = -self.max_velocity
    self.xy_vels_grid[0, 5, 3:12, :] = self.max_velocity
    self.xy_vels_grid[0, 5:13, 13, 1] = self.max_velocity
    self.xy_vels_grid[0, 5:13, 13, 0] = -self.max_velocity
    self.xy_vels_grid[0, 13, 13:19, :] = self.max_velocity
    self.xy_vels_grid[0, 13:19, 20, 1] = self.max_velocity
    self.xy_vels_grid[0, 13:19, 20, 0] = -self.max_velocity
    self.xy_vels_grid[0, 20, 8:21, :] = -self.max_velocity
    self.xy_vels_grid[0, 12:21, 6, 1] = -self.max_velocity
    self.xy_vels_grid[0, 12:21, 6, 0] = self.max_velocity
    self.xy_vels_grid[0, 11, 4:7, :] = -self.max_velocity
    self.xy_vels_grid[0, 11:21, 3, 1] = self.max_velocity
    self.xy_vels_grid[0, 11:21, 3, 0] = -self.max_velocity
    self.xy_vels_grid[0, 21, 3:21, :] = self.max_velocity
    self.xy_vels_grid[0, 12:22, 21, 1] = -self.max_velocity
    self.xy_vels_grid[0, 13:22, 21, 0] = self.max_velocity
    self.xy_vels_grid[0, 3:11, 19, 1] = -self.max_velocity
    self.xy_vels_grid[0, 3:11, 19, 0] = self.max_velocity