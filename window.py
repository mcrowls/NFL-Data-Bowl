class Window:
    def __init__(self, points, optimal_time,optimal_point,triangle = []):
        self.points = points
        self.optimal_time = optimal_time
        self.optimal_point = optimal_point
        self.triangle = triangle
        self.neighbors = [] #a list of windows
        self.g = 0
        self.h = 0
        self.f = 0