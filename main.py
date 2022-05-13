import cv2
import Cards
import numpy as np
import os
import time
import itertools

# Load the train rank and suit images
path = os.path.dirname(os.path.abspath(__file__))
train_suits = Cards.load_suits(path + '/cards_imgs/')

# add camera
print("Define video capture object")
vid = cv2.VideoCapture(0)

# camera output
cam = 10

# framerate calculation
frame_rate_calc = 1
freq = cv2.getTickFrequency()

## Define font to use
font = cv2.FONT_HERSHEY_SIMPLEX

# infinite loop
while True:
    # framerate calculation
    t1 = cv2.getTickCount()

    # Get webcam images
    ret, frame = vid.read()

    # Get height and width of webcam frame
    height, width = frame.shape[:2]

    # Pre-process camera image (gray, blur, and threshold it)
    pre_proc = Cards.preprocess_image(frame)

    # Find and sort the contours of all cards in the image (query cards)
    cnts_sort, cnt_is_card = Cards.find_cards(pre_proc)
    #print('number of cards:', np.sum(cnt_is_card))

    cards = []
    k = 0
    output_image = frame.copy()
    for i in range(len(cnt_is_card)):
        if cnt_is_card[i]:
            cards.append(Cards.preprocess_card(cnts_sort[i], frame))
            cards[k].best_suit_match_diff, cards[k].best_num, cards[k].best_suit_match, cards[k].best_fill_match, cards[k].best_color = Cards.match_card(cards[k], train_suits)
            output_image = Cards.draw_results(output_image, cards[k])
            k += 1

    if len(cards) > 0:
        cv2.imshow('card',cards[0].warp)



    cv2.putText(output_image, "FPS: " + str(int(frame_rate_calc)), (10, 26), font, 0.7, (255, 0, 255), 2, cv2.LINE_AA)

    # calculate set
    single_card_set = set()
    if len(cards) >= 1:
        simple_cards = []
        for index, card in zip(range(len(cards)),cards):
            new_simple_card = Cards.SimpleCard(card, index)
            if new_simple_card.valid:
                simple_cards.append(new_simple_card)
        cv2.putText(output_image, f'Valid cards: {len(simple_cards)}', (10, 75), font, 0.7, (255, 0, 255), 2, cv2.LINE_AA)

        count = 0
        for card_set in itertools.product(simple_cards,simple_cards,simple_cards):
            #print(card_set[0].index, card_set[1].index, card_set[2].index)
            #if card_set[0].index == 0 and card_set[1].index == 1 and card_set[2].index == 2:
            #    print("-"*20)
            #    print(card_set[0])
            #    print(card_set[1])
            #    print(card_set[2])
            #    print("-" * 20)
            count += 1
            if Cards.check_set(card_set):
                print('set_found!', card_set[0].index, card_set[1].index, card_set[2].index)
                single_card_set = set([card_set[0].index,card_set[1].index,card_set[2].index])
                cv2.putText(output_image, "Set found!", (10, 50), font, 0.7, (255, 0, 255), 2, cv2.LINE_AA)
        #cv2.putText(output_image, f'card combinations: {count}', (10, 100), font, 0.7, (255, 0, 255), 2, cv2.LINE_AA)

    # draw card contours
    if len(cards) > 0:
        temp_cnts = []
        temp_set = []
        for i in range(len(cards)):
            if i in single_card_set:
                temp_set.append(cards[i].contour)
            else:
                temp_cnts.append(cards[i].contour)
        cv2.drawContours(output_image, temp_cnts, -1, (0, 0, 255), 2)
        cv2.drawContours(output_image, temp_set, -1, (0, 255, 0), 2)


    if cam == 1:
        cv2.imshow('original image', frame)
    elif cam == 2:
        cv2.imshow('pre-processed image', pre_proc)
    elif cam == 3:
        cv2.imshow('contours image', output_image)
    else:
        # show all images
        cv2.imshow('original image', frame)
        cv2.imshow('pre-processed image', pre_proc)
        cv2.imshow('contours image', output_image)

    # draw image
    # cv2.imshow('Card detector', frame)
    k = cv2.waitKey(1)
    if k == 13:  # 13 is the Enter Key
        break
    elif k == ord('1'):
        cam = 1
    elif k == ord('2'):
        cam = 2
    elif k == ord('3'):
        cam = 3
    elif k == ord('n'):
        cam = 10

    # framerate calculation
    t2 = cv2.getTickCount()
    time1 = (t2-t1)/freq
    frame_rate_calc = 1/time1
    if time1 < 0.1:
        #print(time1)
        wait_time = 0.1 - time1
        #print(wait_time)
        time.sleep(wait_time)
        frame_rate_calc = 10

vid.release()
cv2.destroyAllWindows()
