from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from matplotlib import animation

class myFig(Figure):
    def __init__(self, *args, **kwargs):
        super(myFig, self).__init__(*args, **kwargs)
        self.ax_phi = self.add_subplot(121, projection='polar')
        self.ax_phi.set_yticklabels([])
        self.ax_phi.set_theta_zero_location('N')
        self.ax_phi.set_theta_direction(-1)
        self.phi_point = self.ax_phi.scatter([], [])
        self.ax_theta = self.add_subplot(122, projection='polar')
        self.ax_theta.set_yticklabels([])
        self.ax_theta.set_thetamin(-90)
        self.ax_theta.set_thetamax(90)
        self.theta_point = self.ax_theta.scatter([], [])

    def update(self, dir):
        self.phi_point.set_offsets([[dir[1], 1]])
        self.theta_point.set_offsets([[dir[2], 1]])