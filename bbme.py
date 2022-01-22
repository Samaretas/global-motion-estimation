import itertools
import math
import numpy as np
import time
import cv2
import argparse
import os
# from utils import makedir, draw_motion_vector
import csv

"""
    Code taken from https://github.com/glee1228/Block_matching_algorithm
"""

# python main.py --prev-frame data/frame/frame1100.jpg --current-frame data/frame/frame1105.jpg

def main(args):

    start = time.time()  # 시작 시간 저장

    # 0~255 범위의 값을 갖는 자료형으로 numpy array 데이터 읽기
    prev_img = cv2.imread(args.prev_frame_path,flags=cv2.IMREAD_GRAYSCALE)
    current_img = cv2.imread(args.current_frame_path,flags=cv2.IMREAD_GRAYSCALE)

    prev_frame = np.array(prev_img, dtype=np.uint8)
    cur_frame = np.array(current_img, dtype=np.uint8)


    BM = Block_matcher(block_size=args.block_size,
                         search_range=args.search_range,
                         pixel_acc=args.pixel_acc,
                         searching_procedure = args.searching_procedure)

    interpolated_frame, motion_field = BM.get_motion_field(prev_frame, cur_frame)

    motion_field_x = motion_field[:, :, 0]
    motion_field_y = motion_field[:, :, 1]

    #import pdb;pdb.set_trace()
    drawed_interpolated_frame = draw_motion_vector(interpolated_frame,motion_field)


    sub_matrix = abs(np.array(interpolated_frame, dtype=float) - np.array(prev_frame, dtype=float))
    sub_image = np.array(sub_matrix, dtype=np.uint8)


    # Peak Signal-to-Noise Ratio of the predicted(interpolated) image
    mse = (np.array(sub_image, dtype=float) ** 2).mean()
    psnr = 10 * math.log10((255 ** 2) / mse)

    print('PSNR: %s dB' % psnr)

    running_time = time.time() - start
    print("running time : %06f sec"% running_time)  # 현재시각 - 시작시간 = 실행 시간

    now = time.localtime()
    now_str = "%04d%02d%02d%02d%02d%02d" %(now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec)
    print("%04d/%02d/%02d %02d:%02d:%02d" % (now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec))
    # 입력된 target과 anchor 저장
    makedir('result')
    result_path = os.path.join('result',now_str)
    makedir(result_path)
    cv2.imwrite(os.path.join(result_path,'prev.png'), prev_frame)
    cv2.imwrite(os.path.join(result_path, 'cur.png'), cur_frame)
    cv2.imwrite(os.path.join(result_path,'interpolated.png'), interpolated_frame)
    cv2.imwrite(os.path.join(result_path,'mv_drawing.png'), drawed_interpolated_frame)
    cv2.imwrite(os.path.join(result_path,'error_image.png'), sub_image)


    f = open(os.path.join(result_path,'result.txt'), mode='wt', encoding='utf-8')
    f.write('TIME : {}\n'.format(now_str))
    f.write('RUNNING TIME : %06f \n' % running_time)
    f.write('PREV FRAME PATH : {}\n'.format(args.prev_frame_path))
    f.write('REFERENCE FRAME PATH : {}\n'.format(args.current_frame_path))
    f.write('FRAME SIZE(H,W) : {}, {}\n'.format(args.frame_height,args.frame_width))
    f.write('BLOCK SIZE : {}\n'.format(args.block_size))
    f.write('PIXEL ACC : {}\n'.format(1 if args.pixel_acc==1 else 0.5))
    f.write('SEARCH RANGE : {}\n'.format(args.search_range))
    f.write('SEARCH PROCEDURE : {}\n'.format('exhaustive search' if args.searching_procedure==1 else 'three step search'))
    f.write('SEARCH CRITERIA : {}\n'.format('SAD'))
    f.write('PSNR : %06f \n' %psnr)
    f.close()

    result_dict = {}
    result_dict['now_time'] = now_str
    result_dict['running_time'] = running_time
    result_dict['prev_frame_path'] = args.prev_frame_path
    result_dict['refer_frame_path'] = args.current_frame_path
    result_dict['frame_height'] = args.frame_height
    result_dict['frame_width'] = args.frame_width
    result_dict['block_size'] = args.block_size
    result_dict['pixel_acc'] = 1 if args.pixel_acc==1 else 0.5
    result_dict['search_range'] = args.search_range
    result_dict['search_procedure'] = 'exhaustive search' if args.searching_procedure==1 else 'three step search'
    result_dict['search_criteria'] = 'SAD'
    result_dict['PSNR']=psnr
    return result_dict



class Block_matcher():
    # 프레임 예측과 motion field를 반환
    '''
    SAD : Sum of absolute difference  S(u,v) = \sum (| u-v |)
    '''
    def __init__(self, block_size, search_range, pixel_acc=1,searching_procedure=1):
        self.height = 0                 # height 초기화
        self.width = 0                  # width 초기화
        self.block_size = block_size           # 이미지를 잘라낼 블록 크기(픽셀)
        self.search_range = search_range       # 검색할 픽셀 주위의 범위(픽셀)
        self.pixel_acc = pixel_acc                         # 1 : no interpolation, 2 : bi-linear interpolation
        self.searching_Procedure = searching_procedure  # 1 : Full search(Exhaustive search) , 2 : Three Step search
        self.interpolated_frame = np.empty((self.height, self.width), dtype=np.uint8)
        self.motion_field = np.empty((int(self.height / self.block_size), int(self.width / self.block_size), 2))

    def exhaustive_search(self, prev_frame,cur_frame):
        # 이전 영상에 있는 모든 블록을 반복
        h = self.height
        w = self.width
        b_size = self.block_size
        s_range = self.search_range
        acc = self.pixel_acc
        last_b_row = 0
        for (b_row, b_col) in itertools.product(range(0, h - (b_size - 1), b_size),
                                                    range(0, w - (b_size - 1), b_size)):

            # 이전 프레임에서 일치하는 항목을 검색할 블록
            blk = prev_frame[b_row:b_row + b_size, b_col:b_col + b_size]

            # 이전 프레임 블록과 레퍼런스 블록 사이의 norm의 최소값
            norm_blk_n_min = np.infty

            # 주변 영역에서 SAD norm을 최소화하는 블록을 검색
            for (r_col, r_row) in itertools.product(range(-s_range,(s_range + b_size)),
                                                    range(-s_range, (s_range + b_size))):
                # 후보 블록 왼쪽 상단 정점 및 오른쪽 하단 정점 위치(row, col)
                upper_left_y = (b_row + r_row) * acc
                upper_left_x = (b_col + r_col) * acc
                lower_right_y = (b_row + r_row + b_size - 1) * acc
                lower_right_x = (b_col + r_col + b_size - 1) * acc

                # 앵커 프레임 밖은 검색하지 않음
                if upper_left_y < 0 or upper_left_x < 0 or \
                        lower_right_y > h * acc - 1 or lower_right_x > w * acc - 1:
                    continue

                candidate_blk = cur_frame[upper_left_y:lower_right_y+1,upper_left_x:lower_right_x+1][::acc, ::acc]

                assert candidate_blk.shape == (b_size, b_size)
                #print("({}, {}) | candidate_blk : ({}, {}) | blk : ({}, {}) | search window : ({}, {})".format(blk_row, blk_col,candidate_blk.shape[0],candidate_blk.shape[1],blk.shape[0],blk.shape[1],r_col,r_row))

                diff_prev_cur = np.array(candidate_blk, dtype=np.float16) - np.array(blk, dtype=np.float16)

                norm_blk = np.sum(np.abs(diff_prev_cur))

                # 더 차이가 적은 블록이 발견되면, 모션 벡터 및 블록 저장
                if norm_blk < norm_blk_n_min:
                    norm_blk_n_min = norm_blk
                    matching_blk = candidate_blk
                    dy = r_col
                    dx = r_row

            # 해당 블록과 일치하는 블록으로 예측(보간된) 이미지 생성
            self.interpolated_frame[b_row:b_row + b_size, b_col:b_col + b_size] = matching_blk

            # print("Motion Vector at ({}, {}) : ({},{})".format(b_row/b_size,b_col / b_size, dx, dy))

            # 해당 블록의 각 방향의 모션 벡터
            # import pdb;pdb.set_trace()
            self.motion_field[int(b_row / b_size), int(b_col / b_size), 1] = dx
            self.motion_field[int(b_row / b_size), int(b_col / b_size), 0] = dy

            last_b_row = b_row

        self.interpolated_frame[last_b_row:, : ] = prev_frame[last_b_row:, : ]   # 생성한 이미지에서 채우지 못한 나머지 이미지를 이전 프레임에서 채우기

    def threestep_search(self, prev_frame,cur_frame):
        h = self.height
        w = self.width
        b_size = self.block_size
        s_range = self.search_range
        acc = self.pixel_acc
        last_b_row = 0
        offset1 = int(((2*s_range)+b_size)/3)
        offset2 = int(((2*s_range)+b_size)/5)
        offset3 = int(((2*s_range)+b_size)/10)
        dx,dy = 0,0

        for (b_row, b_col) in itertools.product(range(0, h - (b_size - 1), b_size),
                                                range(0, w - (b_size - 1), b_size)):

            blk = prev_frame[b_row:b_row + b_size, b_col:b_col + b_size]
            norm_blk_n_min = np.infty

            # first step
            for (r_col, r_row) in itertools.product([-offset1,0,offset1],
                                                    [-offset1,0,offset1]):
                # print(r_col,r_row)

                upper_left_y = (b_row + r_row) * acc
                upper_left_x = (b_col + r_col) * acc
                lower_right_y = (b_row + r_row + b_size - 1) * acc
                lower_right_x = (b_col + r_col + b_size - 1) * acc

                if upper_left_y < 0 or upper_left_x < 0 or \
                        lower_right_y > h * acc - 1 or lower_right_x > w * acc - 1:
                    continue

                candidate_blk = cur_frame[upper_left_y:lower_right_y+1,upper_left_x:lower_right_x+1][::acc, ::acc]
                assert candidate_blk.shape == (b_size, b_size)
                # print("({}, {}) | candidate_blk : ({}, {}) | blk : ({}, {}) | search window : ({}, {})".format(blk_row, blk_col,candidate_blk.shape[0],candidate_blk.shape[1],blk.shape[0],blk.shape[1],r_col,r_row))
                diff_prev_cur = np.array(candidate_blk, dtype=np.float16) - np.array(blk, dtype=np.float16)

                norm_blk = np.sum(np.abs(diff_prev_cur))

                if norm_blk < norm_blk_n_min:
                    norm_blk_n_min = norm_blk
                    matching_blk = candidate_blk
                    dy = r_col
                    dx = r_row

            global_dy = dy
            global_dx = dx
            b_row2 = b_row+dx
            b_col2 = b_col+dy
            #print('first step : ({}, {}) ==> ({}, {})'.format(b_row,b_col,b_row2 , b_col2 ))

            for (r_col, r_row) in itertools.product([-offset2, 0, offset2],
                                                    [-offset2, 0, offset2]):

                upper_left_y = (b_row2 + r_row) * acc
                upper_left_x = (b_col2 + r_col) * acc
                lower_right_y = (b_row2 + r_row + b_size - 1) * acc
                lower_right_x = (b_col2 + r_col + b_size - 1) * acc

                if upper_left_y < 0 or upper_left_x < 0 or \
                        lower_right_y > h * acc - 1 or lower_right_x > w * acc - 1:
                    continue

                candidate_blk = cur_frame[upper_left_y:lower_right_y+1,upper_left_x:lower_right_x+1][::acc, ::acc]
                assert candidate_blk.shape == (b_size, b_size)
                # print("({}, {}) | candidate_blk : ({}, {}) | blk : ({}, {}) | search window : ({}, {})".format(blk_row, blk_col,candidate_blk.shape[0],candidate_blk.shape[1],blk.shape[0],blk.shape[1],r_col,r_row))
                diff_prev_cur = np.array(candidate_blk, dtype=np.float16) - np.array(blk, dtype=np.float16)

                norm_blk = np.sum(np.abs(diff_prev_cur))

                if norm_blk < norm_blk_n_min:
                    norm_blk_n_min = norm_blk
                    matching_blk = candidate_blk
                    dy = r_col
                    dx = r_row

            global_dy += dy
            global_dx += dx
            b_row3 = b_row2 + dx
            b_col3 = b_col2 + dy
            #print('second step : ({}, {}) ==> ({}, {})'.format(b_row2,b_col2,b_row3 , b_col3 ))
            for (r_col, r_row) in itertools.product([-offset3, 0, offset3],
                                                    [-offset3, 0, offset3]):

                # print(r_col,r_row)
                upper_left_y = (b_row3 + r_row) * acc
                upper_left_x = (b_col3 + r_col) * acc
                lower_right_y = (b_row3 + r_row + b_size - 1) * acc
                lower_right_x = (b_col3 + r_col + b_size - 1) * acc


                if upper_left_y < 0 or upper_left_x < 0 or \
                        lower_right_y > h * acc - 1 or lower_right_x > w * acc - 1:
                    continue

                candidate_blk = cur_frame[upper_left_y:lower_right_y+1,upper_left_x:lower_right_x+1][::acc, ::acc]
                assert candidate_blk.shape == (b_size, b_size)
                # print("({}, {}) | candidate_blk : ({}, {}) | blk : ({}, {}) | search window : ({}, {})".format(blk_row, blk_col,candidate_blk.shape[0],candidate_blk.shape[1],blk.shape[0],blk.shape[1],r_col,r_row))
                diff_prev_cur = np.array(candidate_blk, dtype=np.float16) - np.array(blk, dtype=np.float16)

                norm_blk = np.sum(np.abs(diff_prev_cur))

                if norm_blk < norm_blk_n_min:
                    norm_blk_n_min = norm_blk
                    matching_blk = candidate_blk
                    dy = r_col
                    dx = r_row
            global_dy += dy
            global_dx += dx

           #print('third step : ({}, {}) ==> ({}, {})'.format(b_row3,b_col3, b_row3+dx, b_col3+dy))

            self.interpolated_frame[b_row:b_row + b_size, b_col:b_col + b_size] = matching_blk

            #print("motion vector at ({}, {}) ==> ({},{})".format(b_row , b_col , global_dx, global_dy))



            # import pdb;pdb.set_trace()
            self.motion_field[int(b_row / b_size), int(b_col / b_size), 1] = dx
            self.motion_field[int(b_row / b_size), int(b_col / b_size), 0] = dy

            last_b_row = b_row

        self.interpolated_frame[last_b_row:, :] = prev_frame[last_b_row:, :]  # 생성한 이미지에서 채우지 못한 나머지 이미지를 이전 프레임에서 채우기


    def get_motion_field(self, prev_frame, cur_frame):
        self.height = prev_frame.shape[0]
        self.width = prev_frame.shape[1]
        self.interpolated_frame = np.empty((self.height, self.width), dtype=np.uint8)
        self.motion_field = np.empty((int(self.height / self.block_size), int(self.width / self.block_size), 2))

        if self.pixel_acc == 1:
            # print('original image is selected')
            pass
        elif self.pixel_acc == 2:
            cur_frame = cv2.resize(cur_frame, dsize=(self.width * 2, self.height * 2), interpolation=cv2.INTER_LINEAR)
            print('bi-linear interpolated image is selected')
        else:
            raise ValueError('origin pixel accuracy = 1 or half-pixel accuracy = 2')


        if self.searching_Procedure == 1:
            self.exhaustive_search(prev_frame,cur_frame)
            print('exhaustive search')

        elif self.searching_Procedure == 2:
            self.threestep_search(prev_frame,cur_frame)
            # print('three step search')

        return self.interpolated_frame, self.motion_field

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--prev-frame', dest='prev_frame_path', required=False, type=str,
                        help='이전 프레임 경로',
                        default='D:\\BlockMatching_video_intp\\data\\frame\\frame1226.jpg')
    parser.add_argument('--current-frame', dest='current_frame_path', required=False, type=str,
                        help='이전 프레임을 예측하는데 사용하는 현재 프레임 경로',
                        default='D:\\BlockMatching_video_intp\\data\\frame\\frame1228.jpg')
    parser.add_argument('--frame-width', dest='frame_width', required=False,
                        help='고정 프레임 너비',default=640)
    parser.add_argument('--frame-height', dest='frame_height', required=False,
                        help='고정 프레임 높이',default=360)
    parser.add_argument('--block-size', dest='block_size', required=False,
                        help='이미지를 잘라낼 블록 크기(픽셀).',default=8)
    parser.add_argument('--search-range', dest='search_range', required=False,
                        help="검색할 픽셀 주위의 범위.",default=7)
    parser.add_argument('--pixel-accuracy', dest='pixel_acc',  default=1, required=False,
                        help="1: Integer-Pel Accuracy (no interpolation), "
                             "2: Half-Pel Integer Accuracy (Bilinear interpolation")
    parser.add_argument('--searching-procedure', dest='searching_procedure', default=1, required=False,
                        help="1: Exhaustive search, "
                             "2: Three Step search")
    args = parser.parse_args()

    block_size_list = [8,16,32]
    search_range_list = [7,15,23]
    pixel_acc_list = [1]
    searching_procedure_list = [1,2]
    total_results= []
    for block_size in block_size_list:
        for search_range in search_range_list:
            for pixel_acc in pixel_acc_list:
                for searching_procedure in searching_procedure_list:
                    args.block_size = block_size
                    args.search_range = search_range
                    args.pixel_acc = pixel_acc
                    args.searching_procedure = searching_procedure
                    result_dict = main(args)
                    total_results.append(result_dict)

    f = open('output.csv', 'w', newline='\n',encoding='utf-8')
    wr = csv.writer(f)
    wr.writerow(total_results[0].keys())
    for i,result in enumerate(total_results):
        wr.writerow(result.values())

    print('finish')
