MAP_1 = {
    # Bedroom 1
    (0, 0): [
        (150, 150, 30), (100, 200, 40), (200, 50, 50),
        (100, 100, 30), (400, 100, 40), (450, 250, 30),
        (100, 400, 40), (200, 350, 30), (350, 400, 50)
    ],
    # Bedroom 2
    (0, 600): [
        (50, 650, 30), (150, 700, 40), (250, 650, 50),
        (350, 750, 30), (400, 850, 40), (450, 700, 30),
        (100, 900, 40), (200, 850, 30), (350, 900, 50)
    ],
    # Kitchen
    (600, 0): [
        (700, 50, 30), (750, 150, 40), (850, 50, 50),
        (900, 300, 30), (700, 250, 40), (800, 100, 30),
        (950, 200, 40), (850, 300, 30), (750, 350, 50)
    ],
    # Living Room
    (600, 600): [
        (750, 700, 30), (700, 750, 40), (750, 850, 50),
        (850, 700, 30), (950, 750, 40), (700, 650, 30),
        (800, 850, 40), (900, 700, 30), (950, 850, 50)
    ],
    # Entrance
    (600, 1200): [
        (750, 1350, 30), (750, 1350, 40), (850, 1300, 50),
        (900, 1400, 30), (700, 1350, 40), (800, 1300, 30),
        (750, 1450, 40), (850, 1400, 30), (650, 1450, 50)
    ],
    # Bathroom
    (1200, 0): [
        (1350, 50, 30), (1350, 100, 40), (1400, 200, 50),
        (1300, 300, 30), (1450, 150, 40), (1500, 100, 30),
        (1350, 250, 40), (1450, 50, 30), (1400, 300, 50)
    ],
    # Toilet
    (1200, 600): [
        (1300, 650, 30), (1350, 750, 40), (1450, 650, 50),
        (1300, 850, 30), (1400, 700, 40), (1500, 750, 30),
        (1350, 850, 40), (1450, 800, 30), (1300, 700, 50)
    ],
}

DOORS = {
    # Bedroom 1 to Kitchen
    (600, 150, 600, 300): 'Kitchen',
    # Bedroom 2 to Living Room
    (600, 750, 600, 900): 'Living Room',
    # Kitchen to Living Room
    (800, 600, 950, 600): 'Living Room',
    # Entrance to Living Room
    (800, 1200, 950, 1200): 'Entrance',
    # Living Room to Toilet
    (1200, 750, 1200, 900): 'Toilet',
    # Kitchen to Bathroom
    (1200, 150, 1200, 300): 'Bathroom',
}
