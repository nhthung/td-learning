####################################################################################################
# watermaze module
####################################################################################################
class watermaze(object):
    
    """
    This class defines a set of functions for simulating a rat moving in a water-maze.
    
    For the purposes of this assignment, you should be using the move function to 
    determine the next state of the environment at each time-step of the simulation.
    
    See the demo of its usage after the module code.
    """
    
    ####################################################################
    # the initialization function, measurements are in cm
    def __init__(self, pool_radius=60, platform_radius=10, platform_location=np.array([25, 25]), 
                 stepsize=5.0, momentum=0.2, T=60):
        
        """
        The init function for the watermaze module.
        
        - The pool_radius argument specifies the radius of the pool.
        
        - The platform_radius argument specifies the radius of the platform.
        
        - The platform_location argument specifies the location of the platform centre.
        
        - The stepsize argument specifies how far the rat moves in one step.
        
        - The momentum argument specifies the ratio of old movement to new movement direction (i.e. 
        momentum = 0 means all new movement, momentum = 1 means all old movement, otherwise a mix.
        
        - The T argument is the maximum time for a trial in the pool.

        
        """
        
        # store the given info
        self.radius            = pool_radius
        self.platform_radius   = platform_radius
        self.platform_location = platform_location
        self.stepsize          = stepsize
        self.momentum          = momentum
        self.T                 = T
        
        # a dictionary for calculating directions
        self.direction = {
            0:  np.pi/2,   #north
            1:  np.pi/4,   #north-east
            2:  0,         #east
            3:  7*np.pi/4, #south-east
            4:  3*np.pi/2, #south
            5:  5*np.pi/4, #south-west
            6:  np.pi,     #west
            7:  3*np.pi/4, #north-west
        }
        
        # initialize the dynamic variables
        self.position = np.zeros((2,T))
        self.t        = 0
        self.prevdir  = np.zeros((2,))
        
    ####################################################################
    # for updating the rat's position in the pool
    def move(self, A, angle=None):

        """
        Updates the simulated rat's position in the water-maze environment by moving it in the 
        specified direction. 
        
        - The argument A is the last selected action, and must be an integer from 0-7, with 0 indicating N, 
        1 indicating NE, etc. 

        """
        
        # check the A argument
        if np.isin(A, np.arange(8)):
            angle = self.direction[A]
        elif A != 8:
            raise ValueError('Error: The argument A must be an integer from 0-8, indicating which action was selected.')
        
        # determine the vector of direction of movement
        newdirection = np.array([np.cos(angle), np.sin(angle)])
        
        # add in momentum to reflect actual swimming dynamics (and normalize, then multiply by stepsize)
        direction = (1.0 - self.momentum)*newdirection + self.momentum*self.prevdir
        direction = direction/np.sqrt((direction**2).sum())
        direction = direction*self.stepsize
        
        # update the position, prevent the rat from actually leaving the water-maze by having it "bounce" off the wall 
        [newposition, direction] = self.poolreflect(self.position[:,self.t] + direction)

        # if we're now at the very edge of the pool, move us in a little-bit
        if (np.linalg.norm(newposition) == self.radius):
            newposition = np.multiply(np.divide(newposition,np.linalg.norm(newposition)),(self.radius - 1))

        # update the position, time (and previous direction)
        self.position[:,self.t+1] = newposition
        self.t                    = self.t + 1
        self.prevdir              = direction
        
        return newposition
        
    ####################################################################
    # for bouncing the rat off the wall of the pool
    def poolreflect(self, newposition):
        
        """
        The poolreflect function returns the point in space at which the rat will be located if it 
        tries to move from the current position to newposition but bumps off the wall of the pool. 
        If the rat would not bump into the wall, then it simply returns newposition. The function 
        also returns the direction the rat will be headed.
        """

        # determine if the newposition is outside the pool, if not, just return the new position
        if (np.linalg.norm(newposition) < self.radius):
            refposition  = newposition
            refdirection = newposition - self.position[:,self.t]

        else:

            # determine where the rat will hit the pool wall
            px = self.intercept(newposition)
            
            # get the tangent vector to this point by rotating -pi/2
            tx = np.asarray(np.matmul([[0, 1], [-1, 0]],px))

            # get the vector of the direction of movement
            dx = px - self.position[:,self.t]
            
            # get the angle between the direction of movement and the tangent vector
            theta = np.arccos(np.matmul((np.divide(tx,np.linalg.norm(tx))).transpose(),(np.divide(dx,np.linalg.norm(dx))))).item()

            # rotate the remaining direction of movement vector by 2*(pi - theta) to get the reflected direction
            ra = 2*(np.pi - theta)
            refdirection = np.asarray(np.matmul([[np.cos(ra), -np.sin(ra)], [np.sin(ra), np.cos(ra)]],(newposition - px)))

            # get the reflected position
            refposition = px + refdirection

        # make sure the new position is inside the pool
        if (np.linalg.norm(refposition) > self.radius):
            refposition = np.multiply((refposition/np.linalg.norm(refposition)),(self.radius - 1))

        return [refposition, refdirection]
    
    ####################################################################
    # for checking when/where the rat hits the edge of the pool
    def intercept(self,newposition):
        
        """
        The intercept function returns the point in space at which the rat will intercept with the pool wall 
        if it is moving from point P1 to point P2 in space, given the pool radius.
        """
        
        # for easy referencing, set p1 and p2
        p1 = self.position[:,self.t]
        p2 = newposition

        # calculate the terms used to find the point of intersection
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        dr = np.sqrt(np.power(dx,2) + np.power(dy,2))
        D  = p1[0]*p2[1] - p2[0]*p1[1]
        sy = np.sign(dy)
        if (sy == 0):
            sy = 1.0
            
        # calculate the potential points of intersection
        pp1 = np.zeros((2,))
        pp2 = np.zeros((2,))

        pp1[0] = (D*dy + sy*dx*np.sqrt((np.power(self.radius,2))*(np.power(dr,2))-np.power(D,2)))/(np.power(dr,2))
        pp2[0] = (D*dy - sy*dx*np.sqrt((np.power(self.radius,2))*(np.power(dr,2))-np.power(D,2)))/(np.power(dr,2))
        pp1[1] = (-D*dx + np.absolute(dy)*np.sqrt((np.power(self.radius,2))*(np.power(dr,2))-np.power(D,2)))/(np.power(dr,2))
        pp2[1] = (-D*dx - np.absolute(dy)*np.sqrt((np.power(self.radius,2))*(np.power(dr,2))-np.power(D,2)))/(np.power(dr,2))

        # determine which intersection point is actually the right one (whichever is closer to p2)
        if np.linalg.norm(p2 - pp1) < np.linalg.norm(p2 - pp2):
            px = pp1

        else:
            px = pp2
        
        return px
    
    ####################################################################
    # sets the start position of the rat in the pool
    def startposition(self):

        # select a random location from the main cardinal axes and calculate its vector angle
        condition = 2*np.random.randint(0,4)
        angle = self.direction[condition]

        # print(np.asarray([np.cos(angle), np.sin(angle)]) * (self.radius - 1))

        self.position[:,0] = np.asarray([np.cos(angle), np.sin(angle)]) * (self.radius - 1)
        # left = np.asarray([-59, 0])
        # self.position[:,0] = left
        
    ####################################################################
    # plot the most recent path of the rat through the pool
    def plotpath(self):
        
        # create the figure 
        fig = plt.figure()
        ax = fig.gca()

        # create the pool perimeter
        pool_perimeter = plt.Circle((0, 0), self.radius, fill=False, color='b', ls='-')
        ax.add_artist(pool_perimeter)

        # create theplatform
        platform = plt.Circle(self.platform_location, self.platform_radius, fill=False, color='r', ls='-')
        ax.add_artist(platform)

        # plot the path
        plt.plot(self.position[0,0:self.t],self.position[1,0:self.t], color='k', ls='-')

        # plot the final location and starting location
        plt.plot(self.position[0,0],self.position[1,0],color='b', marker='o',markersize=4, markerfacecolor='b')
        plt.plot(self.position[0,self.t-1],self.position[1,self.t-1],color='r', marker='o',markersize=6, markerfacecolor='r')

        # adjust the axis
        ax.axis('equal')
        ax.set_xlim((-self.radius-50, self.radius+50))
        ax.set_ylim((-self.radius-50, self.radius+50))
        plt.xticks(np.arange(-self.radius, self.radius+20, step=20))
        plt.yticks(np.arange(-self.radius, self.radius+20, step=20))
        ax.set_xlabel('X Position (cm)')
        ax.set_ylabel('Y Position (cm)')

        # turn on the grid
        plt.grid(True)
        plt.tight_layout()

        # show the figure
        # plt.show()

        return plt, ax
        
    ####################################################################
    # checks whether the time is up
    def timeup(self):
        
        """
        Returns true if the time for the trial is finished, false otherwise.
        """
        
        return self.t > (self.T - 2)
    
    ####################################################################
    # checks whether the rat has found the platform
    def atgoal(self):
        
        """
        Returns true if the rat is on the platform, false otherwise.
        """
        
        return np.sqrt(np.sum((self.position[:,self.t] - self.platform_location)**2)) <= (self.platform_radius + 1)

class Environment:
    '''
    At any given time t, the environment provides the animal with a reward Rt.
    Reward always 0 except when rat reaches platform, then it's 1.
    '''
    def __init__(self, agent):
        self.maze = watermaze()
        agent.maze = self.maze
        agent.S    = sunflower(agent.n, agent.maze.radius) # Init place fields centers
        self.agent = agent

        self.startposition()
    
    def reset(self):
        self.maze.position = np.zeros((2, self.maze.T))
        self.maze.t        = 0
        self.maze.prevdir  = np.zeros((2,))

        self.startposition()
    
    def startposition(self):
        self.maze.startposition()
        self.agent.reset()

    def plot(self, fields=False):
        plt, ax = self.maze.plotpath()

        if fields:
            for x, y in self.agent.S:
                plt.plot(x, y, color='b', marker='+',markersize=1, markerfacecolor='b')
                place_field = plt.Circle((x,y), self.agent.sigma/2, fill=False, color='g', ls='-', lw=0.2)
                ax.add_artist(place_field)
        plt.show()

class ActorCritic:
    def __init__(self,
                #  n     = 177,      # Number of place cells
                #  sigma = 9.6,     # Place field breadth
                 n        = 250,   # Number of place cells
                 sigma    = 5,     # Place field breadth
                 discount = 0.9,   # Discount factor
                 etaa     = 0.01,  # Actor learning rate
                 etac     = 0.03): # Critic learning rate
        
        self.maze = None
        self.S    = None # Place fields centers

        self.PC = np.zeros(n)      # Place cells
        self.A  = np.zeros(8)      # Action cells
        self.Z  = np.zeros((8, n)) # Action cells weights
        self.C  = 0                # Critic cell
        self.W  = np.zeros(n)      # Critic cells weights

        self.sigma    = sigma
        self.discount = discount
        self.etaa     = etaa
        self.etac     = etac
        self.n        = n

        self.p = np.zeros(2) # Position
        self.R = 0           # Reward

    def action(self):
        return np.random.choice(self.A.shape, 1, p=self.P())[0]

    def move(self, a):
        # Save terms to compute weight updates
        C_prev = self.Cp()
        PC_prev = self.fp()
        self.PC = PC_prev

        # Move the rat
        self._move(a)

        # Update weights
        self.update(a, C_prev, PC_prev)
        self.PC = self.fp()
    
    def _move(self, a):
        new_position = self.maze.move(a)
        self.p = new_position
        self.R = int(self.maze.atgoal())
    
    def update(self, a, C_prev, PC_prev):
        '''
        Update weights
        '''
        if self.R == 0:
            # Reward = 0 => not yet reached goal
            d  = self.discount*self.Cp() - C_prev # Prediction error
            dW = self.etac * d * PC_prev
            self.Z += self.dZ(a, d, PC_prev) # Update action cells weights
        else:
            # Reward == 1 => reached goal
            d = self.R - self.C
            dW = self.etac * d * self.fp()

        self.W += dW # Update critic cells weights
    
    def P(self):
        '''
        Actual swimming direction is chosen stochastically with probabilities P
        related to action cell activity
        '''
        A = self.Ap()
        P = np.exp(2*A)/np.sum(np.exp(2*A))
        return P
    
    def Ap(self):
        '''
        Action cells activity.
        Relative preference for swimming in the jth direction at location p
        '''
        self.A = self.fp().dot(self.Z.T)
        return self.A
    
    def fp(self):
        '''
        Place cells activity
        '''
        p_dist = np.linalg.norm(self.p - self.S, axis=1) 
        # self.PC = np.exp(-p_dist/(2*self.sigma**2))
        # return self.PC
        return np.exp(-p_dist/(2*self.sigma**2))
    
    def Cp(self):
        '''
        Critic cell firing rate
        '''
        self.C = self.W.dot(self.fp())
        return self.C
    
    def dZ(self, a, d, PC):
        g = np.zeros((8,1))
        g[a] = 1
        return self.etaa * d * g.dot(PC.reshape(1, self.n))
    
    def reset(self):
        self.p = self.maze.position[:,0]