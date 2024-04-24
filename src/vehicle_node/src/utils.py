import numpy as np
import matplotlib.pyplot as plt
from scipy.special import comb


def get_cartesian(s, d, mapx, mapy, maps):
    """
    Convert Frenet coordinates (s, d) to Cartesian coordinates (x, y)
    """
    # Find the segment with the closest s value
    prev_wp = np.max(np.where(maps <= s)[0])

    # Calculate the proportion along the segment
    seg_s = (s - maps[prev_wp])
    seg_x = mapx[prev_wp+1] - mapx[prev_wp]
    seg_y = mapy[prev_wp+1] - mapy[prev_wp]

    # Normalize segment vector
    seg_norm = np.sqrt(seg_x**2 + seg_y**2)
    seg_x /= seg_norm
    seg_y /= seg_norm

    # Compute the x, y coordinates
    x = mapx[prev_wp] + seg_s * seg_x + d * -seg_y
    y = mapy[prev_wp] + seg_s * seg_y + d * seg_x

    return x, y

def get_frenet(x, y, mapx, mapy, maps):
    """
    Convert Cartesian coordinates (x, y) to Frenet coordinates (s, d)
    """
    # Calculate the closest waypoint index to the current x, y position
    closest_wp = np.argmin(np.sqrt((mapx - x)**2 + (mapy - y)**2))

    # Compute the vector from the closest waypoint to the x, y position
    dx = x - mapx[closest_wp]
    dy = y - mapy[closest_wp]

    # Compute the vector from the closest waypoint to the next waypoint
    next_wp = (closest_wp + 1) % len(mapx)
    next_dx = mapx[next_wp] - mapx[closest_wp]
    next_dy = mapy[next_wp] - mapy[closest_wp]

    # Normalize next segment vector
    seg_norm = np.sqrt(next_dx**2 + next_dy**2)
    next_dx /= seg_norm
    next_dy /= seg_norm

    # Compute the projection of the x, y vector onto the next segment vector
    proj = dx * next_dx + dy * next_dy

    # Frenet d coordinate
    d = np.sqrt(dx**2 + dy**2 - proj**2)

    # Compute the s coordinate by adding the distance along the road
    # to the distance of the projection
    s = maps[closest_wp] + proj

    return s,d
    
    
def moving_average(data, window_size):
    moving_averages = []
    window_sum = 0
    
    # 처음 window_size만큼의 요소에 대한 합을 구함
    for i in range(window_size):
        window_sum += data[i]
        moving_averages.append(window_sum / (i + 1))
    
    # 이후 요소부터는 이동평균을 계산하여 리스트에 추가
    for i in range(window_size, len(data)):
        window_sum += data[i] - data[i - window_size]
        moving_averages.append(window_sum / window_size)
    
    return moving_averages

def check_intersection(p1, p2, p3, p4):
    """ 
    Check if line segments (p1, p2) and (p3, p4) intersect. 
    Returns True if they intersect.
    """
    def ccw(A, B, C):
        return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

    return ccw(p1, p3, p4) != ccw(p2, p3, p4) and ccw(p1, p2, p3) != ccw(p1, p2, p4)

def segment_intersection(p1, p2, p3, p4):
    """ 
    Find the intersection point of line segments (p1, p2) and (p3, p4) 
    if they intersect.
    """
    det = lambda a, b: a[0] * b[1] - a[1] * b[0]
    xdiff = (p1[0] - p2[0], p3[0] - p4[0])
    ydiff = (p1[1] - p2[1], p3[1] - p4[1])
    div = det(xdiff, ydiff)
    if div == 0:
        return None

    d = (det(p1, p2), det(p3, p4))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return (x, y)

def find_intersections(points1, points2):
    """
    Check for intersections between two lists of points that define two lines.
    Returns a list of intersection points.
    """
    intersections = []
    for i in range(len(points1) - 1):
        for j in range(len(points2) - 1):
            if check_intersection(points1[i], points1[i+1], points2[j], points2[j+1]):
                intersect = segment_intersection(points1[i], points1[i+1], points2[j], points2[j+1])
                if intersect:
                    intersections.append((intersect, (i,j)))
                    
    return intersections

def find_intersections_with_indices(points1, points2):
    """
    Check for intersections between two lists of points that define two lines.
    Returns a list of tuples with intersection points and the indices of the segments.
    """
    intersections = []
    for i in range(len(points1) - 1):
        for j in range(len(points2) - 1):
            if check_intersection(points1[i], points1[i+1], points2[j], points2[j+1]):
                intersect = segment_intersection(points1[i], points1[i+1], points2[j], points2[j+1])
                if intersect:
                    intersections.append((intersect, (i, j)))
                    
    return intersections

