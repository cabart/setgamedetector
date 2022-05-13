############## Playing Card Detector Functions ###############
#
# Author: Evan Juras
# Date: 9/5/17
# Description: Functions and classes for CardDetector.py that perform
# various steps of the card detection algorithm


# Import necessary packages
import numpy as np
import cv2
import math
import time

### Constants ###

# Adaptive threshold levels
BKG_THRESH = 60
CARD_THRESH = 30

# Width and height of card corner, where rank and suit are
CORNER_WIDTH = 32
CORNER_HEIGHT = 84

# Dimensions of rank train images
RANK_WIDTH = 70
RANK_HEIGHT = 125

# Dimensions of suit train images
SUIT_WIDTH = 70
SUIT_HEIGHT = 100

RANK_DIFF_MAX = 2000
SUIT_DIFF_MAX = 700

# Original Values
# CARD_MAX_AREA = 120000
# CARD_MIN_AREA = 25000

CARD_MAX_AREA = 200000
CARD_MIN_AREA = 1000

font = cv2.FONT_HERSHEY_SIMPLEX

class Rectangle:
    def __init__(self, x, y, w, h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.center = (x + w/2, y + h/2)
        self.area = w*h

### Structures to hold query card and train card information ###

class Query_card:
    """Structure to store information about query cards in the camera image."""

    def __init__(self):
        self.contour = []  # Contour of card
        self.width, self.height = 0, 0  # Width and height of card
        self.corner_pts = []  # Corner points of card
        self.center = []  # Center point of card
        self.warp = []  # 200x300, flattened, grayed, blurred image
        self.warp_color = []
        self.rank_img = []  # Thresholded, sized image of card's rank
        self.suit_img = []  # Thresholded, sized image of card's suit
        self.best_rank_match = "Unknown"  # Best matched rank
        self.best_suit_match = "Unknown"  # Best matched suit
        self.best_fill_match = "Unknown"
        self.best_num = 0
        self.best_color = [0,0,0]
        self.rank_diff = 0  # Difference between rank image and best matched train rank image
        self.suit_diff = 0  # Difference between suit image and best matched train suit image


class SimpleCard:
    valid = True
    best_num = 4
    best_fill = 4
    best_color = 4
    best_shape = 4

    def __init__(self, q_card: Query_card, index):
        valid = True
        self.index = index

        # best_fill
        if q_card.best_fill_match == 'outline':
            self.best_fill = 0
        elif q_card.best_fill_match == 'dotted':
            self.best_fill = 1
        elif q_card.best_fill_match == 'filled':
            self.best_fill = 2
        else:
            valid = False

        # best_shape
        if q_card.best_suit_match == 'tilde':
            self.best_shape = 0
        elif q_card.best_suit_match == 'circle':
            self.best_shape = 1
        elif q_card.best_suit_match == 'rectangle':
            self.best_shape = 2
        else:
            valid = False

        # best_num
        if q_card.best_num == 1:
            self.best_num = 0
        elif q_card.best_num == 2:
            self.best_num = 1
        elif q_card.best_num == 3:
            self.best_num = 2
        else:
            valid = False

        # best_color
        if q_card.best_color == [0, 0, 255]:
            self.best_color = 0
        elif q_card.best_color == [0, 255, 0]:
            self.best_color = 1
        elif q_card.best_color == [255, 0, 0]:
            self.best_color = 2
        else:
            valid = False

        self.valid = valid

    def __repr__(self):
        return f'{self.best_num},{self.best_color},{self.best_fill},{self.best_shape}'


class Train_ranks:
    """Structure to store information about train rank images."""

    def __init__(self):
        self.img = []  # Thresholded, sized rank image loaded from hard drive
        self.name = "Placeholder"


class Train_suits:
    """Structure to store information about train suit images."""

    def __init__(self):
        self.img = []  # Thresholded, sized suit image loaded from hard drive
        self.name = "Placeholder"


### Functions ###
def load_ranks(filepath):
    """Loads rank images from directory specified by filepath. Stores
    them in a list of Train_ranks objects."""

    train_ranks = []
    i = 0

    for Rank in ['Ace', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven',
                 'Eight', 'Nine', 'Ten', 'Jack', 'Queen', 'King']:
        train_ranks.append(Train_ranks())
        train_ranks[i].name = Rank
        filename = Rank + '.jpg'
        train_ranks[i].img = cv2.imread(filepath + filename, cv2.IMREAD_GRAYSCALE)
        i = i + 1

    return train_ranks


def load_suits(filepath):
    """Loads suit images from directory specified by filepath. Stores
    them in a list of Train_suits objects."""

    train_suits = []
    i = 0

    # for Suit in ['tilde_outline', 'tilde', 'rectangle_outline', 'rectangle', 'circle', 'circle_outline']:
    for Suit in ['tilde', 'rectangle', 'circle']:
        train_suits.append(Train_suits())
        train_suits[i].name = Suit
        filename = Suit + '.png'
        template_image = cv2.bitwise_not(cv2.imread(filepath + filename, cv2.IMREAD_GRAYSCALE))
        _, train_suits[i].img = cv2.threshold(template_image, 150, 255, cv2.THRESH_BINARY)
        #print("suit size", train_suits[i].img.shape)
        i = i + 1

    return train_suits


def preprocess_image(image):
    """Returns a grayed, blurred, and adaptively thresholded camera image."""

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # The best threshold level depends on the ambient lighting conditions.
    # For bright lighting, a high threshold must be used to isolate the cards
    # from the background. For dim lighting, a low threshold must be used.
    # To make the card detector independent of lighting conditions, the
    # following adaptive threshold method is used.
    #
    # A background pixel in the center top of the image is sampled to determine # TODO: don't like this
    # its intensity. The adaptive threshold is set at 50 (THRESH_ADDER) higher
    # than that. This allows the threshold to adapt to the lighting conditions.
    img_w, img_h = np.shape(image)[:2]
    bkg_level = gray[int(img_h / 100)][int(img_w / 2)]
    thresh_level = bkg_level + BKG_THRESH

    retval, thresh = cv2.threshold(blur, thresh_level, 255, cv2.THRESH_BINARY)

    return thresh


def find_cards(thresh_image):
    """Finds all card-sized contours in a thresholded camera image.
    Returns the number of cards, and a list of card contours sorted
    from largest to smallest."""

    # Find contours and sort their indices by contour size
    cnts, hier = cv2.findContours(thresh_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    index_sort = sorted(range(len(cnts)), key=lambda i: cv2.contourArea(cnts[i]), reverse=True)

    # If there are no contours, do nothing
    if len(cnts) == 0:
        return [], []

    # Otherwise, initialize empty sorted contour and hierarchy lists
    cnts_sort = []
    hier_sort = []
    cnt_is_card = np.zeros(len(cnts), dtype=int)

    # Fill empty lists with sorted contour and sorted hierarchy. Now,
    # the indices of the contour list still correspond with those of
    # the hierarchy list. The hierarchy array can be used to check if
    # the contours have parents or not.
    for i in index_sort:
        cnts_sort.append(cnts[i])
        hier_sort.append(hier[0][i])

    # Determine which of the contours are cards by applying the
    # following criteria: 1) Smaller area than the maximum card size,
    # 2), bigger area than the minimum card size, 3) have no parents,
    # and 4) have four corners

    for i in range(len(cnts_sort)):
        size = cv2.contourArea(cnts_sort[i])
        peri = cv2.arcLength(cnts_sort[i], True)
        approx = cv2.approxPolyDP(cnts_sort[i], 0.01 * peri, True)

        if ((size < CARD_MAX_AREA) and (size > CARD_MIN_AREA)
                and (hier_sort[i][3] == -1) and (len(approx) == 4)):
            cnt_is_card[i] = 1

    return cnts_sort, cnt_is_card


def preprocess_card(contour, image):
    """Uses contour to find information about the query card. Isolates rank
    and suit images from the card."""

    # Initialize new Query_card object
    qCard = Query_card()

    qCard.contour = contour

    # Find perimeter of card and use it to approximate corner points
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.01 * peri, True)
    pts = np.float32(approx)
    qCard.corner_pts = pts

    # Find width and height of card's bounding rectangle
    x, y, w, h = cv2.boundingRect(contour)
    qCard.width, qCard.height = w, h

    # Find center point of card by taking x and y average of the four corners.
    average = np.sum(pts, axis=0) / len(pts)
    cent_x = int(average[0][0])
    cent_y = int(average[0][1])
    qCard.center = [cent_x, cent_y]

    # Warp card into 200x300 flattened image using perspective transform
    qCard.warp, qCard.warp_color = flattener(image, pts, w, h)

    # Grab corner of warped card image and do a 4x zoom
    Qcorner = qCard.warp[0:CORNER_HEIGHT, 0:CORNER_WIDTH]
    Qcorner_zoom = cv2.resize(Qcorner, (0, 0), fx=4, fy=4)

    # Sample known white pixel intensity to determine good threshold level
    white_level = Qcorner_zoom[15, int((CORNER_WIDTH * 4) / 2)]
    thresh_level = white_level - CARD_THRESH
    if (thresh_level <= 0):
        thresh_level = 1
    retval, query_thresh = cv2.threshold(Qcorner_zoom, thresh_level, 255, cv2.THRESH_BINARY_INV)

    # Split in to top and bottom half (top shows rank, bottom shows suit)
    Qrank = query_thresh[20:185, 0:128]
    Qsuit = query_thresh[186:336, 0:128]

    # Find rank contour and bounding rectangle, isolate and find largest contour
    Qrank_cnts, hier = cv2.findContours(Qrank, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    Qrank_cnts = sorted(Qrank_cnts, key=cv2.contourArea, reverse=True)

    # Find bounding rectangle for largest contour, use it to resize query rank
    # image to match dimensions of the train rank image
    if len(Qrank_cnts) != 0:
        x1, y1, w1, h1 = cv2.boundingRect(Qrank_cnts[0])
        Qrank_roi = Qrank[y1:y1 + h1, x1:x1 + w1]
        Qrank_sized = cv2.resize(Qrank_roi, (RANK_WIDTH, RANK_HEIGHT), 0, 0)
        qCard.rank_img = Qrank_sized

    # Find suit contour and bounding rectangle, isolate and find largest contour
    Qsuit_cnts, hier = cv2.findContours(Qsuit, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    Qsuit_cnts = sorted(Qsuit_cnts, key=cv2.contourArea, reverse=True)

    # Find bounding rectangle for largest contour, use it to resize query suit
    # image to match dimensions of the train suit image
    if len(Qsuit_cnts) != 0:
        x2, y2, w2, h2 = cv2.boundingRect(Qsuit_cnts[0])
        Qsuit_roi = Qsuit[y2:y2 + h2, x2:x2 + w2]
        Qsuit_sized = cv2.resize(Qsuit_roi, (SUIT_WIDTH, SUIT_HEIGHT), 0, 0)
        qCard.suit_img = Qsuit_sized

    return qCard


def match_card(q_card: Query_card, train_suits: list[Train_suits]):
    """Finds best rank and suit matches for the query card. Differences
    the query card rank and suit images with the train rank and suit images.
    The best match is the rank or suit image that has the least difference."""

    best_score = 0
    best_shape_name = 'unknown'
    best_fill_name = 'unknown'
    best_color = [0,0,0]

    # find bounding boxes of shapes (circles, tildes, rectangles)

    # create black and white threshold image and find all contours
    _, threshold_img = cv2.threshold(q_card.warp, 180, 255, cv2.THRESH_BINARY_INV)
    qCnts, qHier = cv2.findContours(threshold_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    threshold_img = cv2.cvtColor(threshold_img, cv2.COLOR_GRAY2RGBA)

    # find polygon shapes with correct size and get bounding box
    approx_rectangles = []
    for q in qCnts:
        epsilon = 0.005 * cv2.arcLength(q, True)
        approx = cv2.approxPolyDP(q, epsilon, True)
        x,y,w,h = cv2.boundingRect(approx)
        new_rect = Rectangle(x, y, w, h)
        if 6000 < new_rect.area < 12000:
            approx_rectangles.append(new_rect)

    # filter double rectangles for outlines, keep larger rectangle
    distance_threshold = 5
    correct_rectangles = []
    for i, rect in zip(range(len(approx_rectangles)), approx_rectangles):
        keep = True
        for j, rectTest in zip(range(len(approx_rectangles)), approx_rectangles):
            if i != j:
                dist = math.dist(rect.center, rectTest.center)
                if dist < distance_threshold and rect.area < rectTest.area:
                    keep = False
        if keep:
            correct_rectangles.append(rect)

    # draw correct rectangles
    correct_rectangles_image = threshold_img.copy()
    for rect in correct_rectangles:
        cv2.rectangle(correct_rectangles_image, (rect.x, rect.y), (rect.x + rect.w, rect.y + rect.h), (0, 255, 255), 1)

    num_symbols = len(correct_rectangles)

    # show resulting rectangles
    cv2.putText(correct_rectangles_image, '#shapes: ' + str(num_symbols), (10, 30), font, 1, (255, 0, 0), 1, cv2.LINE_AA)
    cv2.imshow('correct rectangles', correct_rectangles_image)

    # find best shape and fill
    if len(correct_rectangles) > 0:
        sample_rectangle = correct_rectangles[0]
        x = sample_rectangle.x
        y = sample_rectangle.y
        w = sample_rectangle.w
        h = sample_rectangle.h
        sample_threshold_img = threshold_img[y:y+h, x:x+w]
        # test against all templates to find best shape
        for template_shape in train_suits:
            template_shape_resized = cv2.resize(template_shape.img, (w, h))
            template_test = cv2.bitwise_and(sample_threshold_img, sample_threshold_img, mask=template_shape_resized)
            energy_sum = cv2.sumElems(template_test)[0]/10000
            if energy_sum > best_score:
                best_score = energy_sum
                best_shape_name = template_shape.name

            # show score and matching
            cv2.putText(template_test, f'{energy_sum:.2f}', (0, 30), font, 1, (255, 0, 0), 1, cv2.LINE_AA)
            cv2.imshow(f'template_test_{template_shape.name}', template_test)

        # find best fill
        energy_sum = cv2.sumElems(sample_threshold_img)[0]/10000
        if energy_sum < 40:
            best_fill_name = 'outline'
        elif energy_sum < 110:
            best_fill_name = 'dotted'
        else:
            best_fill_name = 'filled'
        cv2.putText(sample_threshold_img, f'{energy_sum:.2f}', (0, 30), font, 1, (0, 255, 0), 3, cv2.LINE_AA)
        cv2.imshow('sample shape', sample_threshold_img)

        # find best color, BGRA format
        color_sample_rectangle = q_card.warp_color[y:y+h, x:x+w]
        energy_sum = cv2.sumElems(color_sample_rectangle)[0:-1]
        max_index = energy_sum.index(max(energy_sum))
        best_color[max_index] = 255
        cv2.imshow('color_card',color_sample_rectangle)
        #print("best_color",energy_sum)
        #print("sample",color_sample_rectangle[10,10])
        #time.sleep(1)

    return best_score, num_symbols, best_shape_name, best_fill_name, best_color


def draw_results(image, qCard):
    """Draw the card name, center point, and contour on the camera image."""

    x = qCard.center[0]
    y = qCard.center[1]
    cv2.circle(image, (x, y), 5, (255, 0, 0), -1)

    #rank_name = qCard.best_rank_match
    fill_name = qCard.best_fill_match
    #suit_name = qCard.best_suit_match + f"{qCard.best_num}, {qCard.best_suit_match_diff:.2f}"
    #suit_name = qCard.best_suit_match + str(qCard.best_suit_match_diff)

    # Draw card name twice, so letters have black outline
    # yellow text (50, 200, 200)
    cv2.putText(image, (f'{qCard.best_num} {qCard.best_suit_match} {fill_name}'), (x - 60, y - 10), font, 1, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(image, (f'{qCard.best_num} {qCard.best_suit_match} {fill_name}'), (x - 60, y - 10), font, 1, qCard.best_color, 2, cv2.LINE_AA)

    #cv2.putText(image, suit_name, (x - 60, y + 25), font, 1, (0, 0, 0), 3, cv2.LINE_AA)
    #cv2.putText(image, suit_name, (x - 60, y + 25), font, 1, (50, 200, 200), 2, cv2.LINE_AA)

    # Can draw difference value for troubleshooting purposes
    # (commented out during normal operation)
    # r_diff = str(qCard.rank_diff)
    # s_diff = str(qCard.suit_diff)
    # cv2.putText(image,r_diff,(x+20,y+30),font,0.5,(0,0,255),1,cv2.LINE_AA)
    # cv2.putText(image,s_diff,(x+20,y+50),font,0.5,(0,0,255),1,cv2.LINE_AA)

    return image


def flattener(image, pts, w, h):
    """Flattens an image of a card into a top-down 200x300 perspective.
    Returns the flattened, re-sized, grayed image.
    See www.pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/"""
    temp_rect = np.zeros((4, 2), dtype="float32")

    s = np.sum(pts, axis=2)

    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]

    diff = np.diff(pts, axis=-1)
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]

    # Need to create an array listing points in order of
    # [top left, top right, bottom right, bottom left]
    # before doing the perspective transform

    if w <= 0.8 * h:  # If card is vertically oriented
        temp_rect[0] = tl
        temp_rect[1] = tr
        temp_rect[2] = br
        temp_rect[3] = bl

    if w >= 1.2 * h:  # If card is horizontally oriented
        temp_rect[0] = bl
        temp_rect[1] = tl
        temp_rect[2] = tr
        temp_rect[3] = br

    # If the card is 'diamond' oriented, a different algorithm
    # has to be used to identify which point is top left, top right
    # bottom left, and bottom right.

    if w > 0.8 * h and w < 1.2 * h:  # If card is diamond oriented
        # If furthest left point is higher than furthest right point,
        # card is tilted to the left.
        if pts[1][0][1] <= pts[3][0][1]:
            # If card is titled to the left, approxPolyDP returns points
            # in this order: top right, top left, bottom left, bottom right
            temp_rect[0] = pts[1][0]  # Top left
            temp_rect[1] = pts[0][0]  # Top right
            temp_rect[2] = pts[3][0]  # Bottom right
            temp_rect[3] = pts[2][0]  # Bottom left

        # If furthest left point is lower than furthest right point,
        # card is tilted to the right
        if pts[1][0][1] > pts[3][0][1]:
            # If card is titled to the right, approxPolyDP returns points
            # in this order: top left, bottom left, bottom right, top right
            temp_rect[0] = pts[0][0]  # Top left
            temp_rect[1] = pts[3][0]  # Top right
            temp_rect[2] = pts[2][0]  # Bottom right
            temp_rect[3] = pts[1][0]  # Bottom left

    maxWidth = 200
    maxHeight = 300

    # Create destination array, calculate perspective transform matrix,
    # and warp card image
    dst = np.array([[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]], np.float32)
    M = cv2.getPerspectiveTransform(temp_rect, dst)
    warp = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    warp_color = warp.copy()
    warp = cv2.cvtColor(warp, cv2.COLOR_BGR2GRAY)

    return warp, warp_color


def check_property(a, b, c):
    property_sum = a + b + c
    if property_sum == 0 or property_sum == 3 or property_sum == 6:
        return True
    else:
        return False


def check_set(cards: list[SimpleCard]):
    ''' list of exactly 3 elements'''
    if cards[0].index == cards[1].index:
        return False

    # fill
    fill = check_property(cards[0].best_fill, cards[1].best_fill, cards[2].best_fill)
    color = check_property(cards[0].best_color, cards[1].best_color, cards[2].best_color)
    shape = check_property(cards[0].best_shape, cards[1].best_shape, cards[2].best_shape)
    number = check_property(cards[0].best_num, cards[1].best_num, cards[2].best_num)
    if fill and color and shape and number:
        return True
    else:
        return False