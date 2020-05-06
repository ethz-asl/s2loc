
class LcCandidate(object):
    def __init__(self, timestamps, clouds):
        self.timestamps = timestamps
        self.clouds = clouds

    def get_clouds(self):
        return self.clouds

    def get_timestamps(self):
        return self.timestamps
