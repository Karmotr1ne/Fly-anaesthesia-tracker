"""Bounded preview raster, accumulating every accepted detection, not display frames."""
import cv2
import numpy as np


class TrajectoryLayer:
    def __init__(self, shape, maximum_width=1080):
        height, width = shape[:2]
        self.scale = min(1., maximum_width / width)
        self.image = np.zeros((max(1, round(height*self.scale)),
                               max(1, round(width*self.scale)), 4), np.uint8)
        self.previous = {}

    def update(self, frame_index, records):
        colours = ((50, 205, 50, 210), (255, 180, 0, 210), (0, 180, 255, 210),
                   (200, 80, 220, 210), (255, 120, 120, 210), (80, 220, 220, 210))
        for record in records:
            chamber = record['chamber_id']
            if (record.get('tracking_status', 'Accepted') != 'Accepted' or
                    not np.isfinite(record['x']) or not np.isfinite(record['y'])):
                self.previous.pop(chamber, None)
                continue
            point = (int(np.clip(round(record['x']*self.scale), 0, self.image.shape[1]-1)),
                     int(np.clip(round(record['y']*self.scale), 0, self.image.shape[0]-1)))
            segment = record.get('segment_id', 0)
            previous = self.previous.get(chamber)
            colour = colours[(int(chamber)-1) % len(colours)]
            if previous and previous[0] == frame_index-1 and previous[1] == segment:
                if previous[2] != point:
                    cv2.line(self.image, previous[2], point, colour, 1, cv2.LINE_AA)
            else:
                cv2.circle(self.image, point, 1, colour, -1)
            self.previous[chamber] = (frame_index, segment, point)

    def snapshot(self):
        return self.image.copy()
