"""Segment Anything auxiliary functions"""

import nuke

MAX_POINTS = 8


def set_default(last_point, visible=False):
    node = nuke.thisNode()
    point = f"point_{last_point:02}"  # point_01

    # Adjust visibility
    node[f"{point}"].setVisible(visible)
    node[f"{point}_l"].setVisible(visible)
    node[f"{point}_e"].setVisible(visible)
    node[f"{point}_s"].setVisible(visible)

    # Set the default values
    node[f"{point}"].clearAnimated()
    node[f"{point}"].setValue((0, (last_point - 1) * 25))

    node[f"{point}_e"].clearAnimated()
    node[f"{point}_e"].setValue(True)

    node[f"{point}_s"].clearAnimated()
    node[f"{point}_s"].setValue(False)


def reset_points():
    # Reset all points
    for i in range(1, MAX_POINTS + 1):
        set_default(i)

    # Keep the first 4 points point visible
    for i in range(1, 5):
        set_default(i, True)


def add_point():
    node = nuke.thisNode()  # Get the current node

    visible_points = 0
    for i in range(1, MAX_POINTS + 1):
        if node["point_{:02}_l".format(i)].visible():
            visible_points += 1

    if visible_points >= MAX_POINTS:
        nuke.tprint("Segment Anything: max 8 points reached")
        return

    # Reveal the next set of knobs
    next_point = visible_points + 1
    set_default(next_point, True)


def remove_point():
    max_points = 8
    node = nuke.thisNode()  # Get the current node

    last_point = 0
    for i in range(1, max_points + 1):
        if node["point_{:02}_l".format(i)].visible():
            last_point = i

    if last_point == 1:
        return

    # Hide the last set of knobs
    set_default(last_point, False)
