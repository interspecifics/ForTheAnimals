"""
Scout
-----
.Genera tonos de llamado random
.Monitorea los sensores IR
.Si detecta una presencia:
    .Inicia modo de percepción/clasificación
    .Si se trata de especimenes de la especie de interés
        .Estampa de tiempo con especie, sonido, video
        .Inicia registro de video 


emmanuel@interspecifics.cc
2020.01.29 // v.ii.x_:x
>> OTRAS NOTAS

https://www.raspberrypi.org/forums/viewtopic.php?t=240200
https://learn.adafruit.com/adafruit-amg8833-8x8-thermal-camera-sensor/raspberry-pi-thermal-camera

check devices
$ v4l2-ctl --list-devices
try recording
$ ffmpeg -i /dev/video7 -vcodec copy capture/cinco.mkv                      # 6.5Mbps sin reencodear

rangos de frecuencias
A- Hz(179-243)
B- Hz(158-174)
C- Hz(142-148)
D- Hz(128-139)
E- Hz(117-124)
F- Hz(106-115)
G- Hz(90-104)

"""

import busio, board, adafruit_amg88xx
import time, argparse, collections, random
import operator, re, os, subprocess
import cv2
import cvtf
import numpy as np
import tflite_runtime.interpreter as tflite
from PIL import Image
from oscpy.client import OSCClient


# minimal temperature difference
MIN_TEMP_DIFF = 2
MIN_MASA_DETEC = 3

# -create objects to communicate the sensor
i2c_bus = busio.I2C(board.SCL, board.SDA)
sensor_a = adafruit_amg88xx.AMG88XX(i2c_bus, 0x68)
sensor_b = adafruit_amg88xx.AMG88XX(i2c_bus, 0x69)
Category = collections.namedtuple('Category', ['id', 'score'])


# img utils
def create_blank(w, h, rgb_color=(0, 0, 0)):
    """ create new image(numpy array) filled with certain color in rgb """
    image = np.zeros((h, w), np.uint8)
    color = tuple(reversed(rgb_color))
    image[:] = 0
    return image

# sensor functions
def read_sensor_pixels(sensor, verbose=False):
    """ Lee los pixeles de temperatura de un sensor
        Devuelve la temperatura media y una lista de temperaturas
        La opción verbose muestra los valores 
    """
    mean_temp = 0
    array_temps = []
    for row in sensor.pixels:
        array_temps.extend(row)
    mean_temp = sum(array_temps)
    mean_temp = mean_temp / len(array_temps)
    if verbose:
        print("\n")
        print ('[Tm]: {0:.2f}'.format(mean_temp))
        for row in sensor.pixels:
            ls = ['{0:.1f}'.format(temp) for temp in row]
            print(' '.join(ls))
        print("\n")
    return mean_temp, array_temps

def dual_detect(verbose=False):
    """ Llama a read_sensor_pixels una vez por cada sensor
        Devuelve el número de celdas ocupadas en cada sensor
        Con verbose muestra paneles de detección
    """
    m_ta, arr_ta = read_sensor_pixels(sensor_a)
    m_tb, arr_tb = read_sensor_pixels(sensor_b)
    na = len(list(filter(lambda x: (x - m_ta) >= MIN_TEMP_DIFF, arr_ta)))
    nb = len(list(filter(lambda x: (x - m_tb) >= MIN_TEMP_DIFF, arr_tb)))
    if verbose:
        print("\n")
        print ('[t1]:{0:.1f}\t[t2]:{1:.1f}'.format(m_tb, m_ta))
        for ix in range(8):
            la = ''.join(['.' if (arr_ta[iy * 8 + ix] - m_ta) < MIN_TEMP_DIFF else '+' for iy in range(8)])
            lb = ''.join(['.' if (arr_tb[iy * 8 + ix] - m_tb) < MIN_TEMP_DIFF else '+' for iy in range(8)])
            print(lb,'\t',la)
        print ('[o1]:{0:d}\t\t[o2]:{1:d}'.format(nb, na))
        print("\n")
    return na, nb

def dual_detect(arg_name, verbose=False):
    """ Llama a read_sensor_pixels una vez por cada sensor
        Devuelve el número de celdas ocupadas en cada sensor (mas los data_sens para log)
        Con verbose muestra paneles de detección
    """
    m_ta, arr_ta = read_sensor_pixels(sensor_a)
    m_tb, arr_tb = read_sensor_pixels(sensor_b)
    na = len(list(filter(lambda x: (x - m_ta) >= MIN_TEMP_DIFF, arr_ta)))
    nb = len(list(filter(lambda x: (x - m_tb) >= MIN_TEMP_DIFF, arr_tb)))
    if verbose:
        print("\n")
        print ('[t{2}]:{0:.1f}\t[t{3}]:{1:.1f}'.format(m_tb, m_ta, arg_name[1], arg_name[0]))
    sens_a = ""
    sens_b = ""
    for ix in range(8):
        la = ''.join(['.' if (arr_ta[iy * 8 + ix] - m_ta) < MIN_TEMP_DIFF else '+' for iy in range(8)])
        lb = ''.join(['.' if (arr_tb[iy * 8 + ix] - m_tb) < MIN_TEMP_DIFF else '+' for iy in range(8)])
        sens_a+=la+'\n'
        sens_b+=lb+'\n'
        if verbose:
            print(lb,'\t',la)
    if verbose:
        print ('[o{2}]:{0:d}\t\t[o{3}]:{1:d}'.format(nb, na, arg_name[1], arg_name[0]))
        print("\n")
    return na, nb, [sens_a, sens_b, m_ta, m_tb]



# detection functions
def load_labels(path):
    p = re.compile(r'\s*(\d+)(.+)')
    with open(path, 'r', encoding='utf-8') as f:
        lines = (p.match(line).groups() for line in f.readlines())
    return {int(num): text.strip() for num, text in lines}

def get_output(interpreter, top_k, score_threshold):
    """Returns no more than top_k categories with score >= score_threshold."""
    scores = cvtf.output_tensor(interpreter, 0)
    categories = [
        Category(i, scores[i])
        for i in np.argpartition(scores, -top_k)[-top_k:]
        if scores[i] >= score_threshold
    ]
    return sorted(categories, key=operator.itemgetter(1), reverse=True)

def append_results_to_img(cv2_im, results, labels):
    height, width, channels = cv2_im.shape
    for ii, res in  enumerate(results):
        percent = int(100 * res.score)
        label = '{}% {}'.format(percent, labels[res[0]])
        cv2_im = cv2.putText(cv2_im, label, (600, 20+ii*30), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 1)
    return cv2_im

def parse_results(cv2_im, results, labels):
    height, width, channels = cv2_im.shape
    for ii, res in  enumerate(results):
        percent = int(100 * res.score)
        label = '{}% {}'.format(percent, labels[res[0]])
        cv2_im = cv2.putText(cv2_im, label, (600, 20+ii*30), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 1)
    return cv2_im



# define callbacks
def human_callback(witch, arg_path, arg_name, arg_recfile, data_sens):
    # choosw from witch
    label = "HUMAN"
    timetag = time.strftime("%Y%m%d_%H%M%S")
    # log to record file
    record_file = open(arg_recfile, 'a+')
    if (witch==1): record_file.write("[scout.detection]: <{0}> <{1}>\n".format(timetag, arg_name[0]))
    elif (witch==2): record_file.write("[scout.detection]: <{0}> <{1}>\n".format(timetag, arg_name[1]))
    record_file.write(">> [label]: {}\n".format(label));
    record_file.write('>> [sensor.name]:{0}\n'.format(arg_name[1]))
    record_file.write('>> [mean_temperature]: {0:.2f} C\n'.format(data_sens[3]))
    record_file.write('>> [data]: \n')
    record_file.write(data_sens[1])
    record_file.write('>> [sensor.name]:{0}\n'.format(arg_name[0]))
    record_file.write('>> [mean_temperature]: {0:.2f} C\n'.format(data_sens[2]))
    record_file.write('>> [data]: \n')
    record_file.write(data_sens[0])
    # cmd exck
    out_filename = ''
    if (witch==1):
        out_filename = arg_path + timetag +"_"+label+"_"+ arg_name[0] +".mkv"
        #cmd = "ffmpeg -i /dev/video6 -vcodec h264_omx -b:v 2M -t 15 " + out_filename
        cmd = "ffmpeg -i /dev/video6 -t 15 -vcodec copy " + out_filename
    elif(witch==2):
        out_filename = arg_path + timetag +"_"+label+"_"+ arg_name[1] +".mkv"
        #cmd = "ffmpeg -i /dev/video2 -vcodec h264_omx -b:v 2M -t 15 " + out_filename
        cmd = "ffmpeg -i /dev/video2 -t 15 -vcodec copy " + out_filename
    else:
        pass
    list_cmd = cmd.split(' ')
    # actualiza y cierra registro
    record_file.write('>> [video.capture]:{0}\n\n'.format(out_filename))
    record_file.close()
    # ejecuta
    cmd_out = subprocess.run(list_cmd, stdout=subprocess.PIPE)
    # print(cmd_out.stdout.decode('utf-8'))
    return cmd_out.stdout.decode('utf-8')

def label_callback(label, witch, arg_path, arg_name, arg_recfile, data_sens):
    # choosw from witch
    #label = "HUMAN"
    timetag = time.strftime("%Y%m%d_%H%M%S")
    # log to record file
    record_file = open(arg_recfile, 'a+')
    if (witch==1): record_file.write("[scout.detection]: <{0}> <{1}>\n".format(timetag, arg_name[0]))
    elif (witch==2): record_file.write("[scout.detection]: <{0}> <{1}>\n".format(timetag, arg_name[1]))
    record_file.write(">> [label]: {}\n".format(label));
    record_file.write('>> [sensor.name]:[{0}]\n'.format(arg_name[1]))
    record_file.write('>> [mean_temperature]: {0:.2f} C\n'.format(data_sens[3]))
    record_file.write('>> [data]: \n')
    record_file.write(data_sens[1])
    record_file.write('>> [sensor.name]:[{0}]\n'.format(arg_name[0]))
    record_file.write('>> [mean_temperature]: {0:.2f} C\n'.format(data_sens[2]))
    record_file.write('>> [data]: \n')
    record_file.write(data_sens[0])
    # cmd exck
    out_filename = ''
    if (witch==1):
        out_filename = arg_path + timetag +"_"+label+"_"+ arg_name[0] +".mkv"
        #cmd = "ffmpeg -i /dev/video6 -vcodec h264_omx -b:v 2M -t 15 " + out_filename
        cmd = "ffmpeg -i /dev/video6 -t 15 -vcodec copy " + out_filename
    elif(witch==2):
        out_filename = arg_path + timetag +"_"+label+"_"+ arg_name[1] +".mkv"
        #cmd = "ffmpeg -i /dev/video2 -vcodec h264_omx -b:v 2M -t 15 " + out_filename
        cmd = "ffmpeg -i /dev/video2 -t 15 -vcodec copy " + out_filename
    else:
        pass
    list_cmd = cmd.split(' ')
    # actualiza y cierra registro
    record_file.write('>> [video.capture]:{0}\n\n\n'.format(out_filename))
    record_file.close()
    # ejecuta
    cmd_out = subprocess.run(list_cmd, stdout=subprocess.PIPE)
    # print(cmd_out.stdout.decode('utf-8'))
    return cmd_out.stdout.decode('utf-8')


# soundsys
def update_soundsystem(arg_recfile, arg_name, osc_c):
    """
    envía mensajes a sc que disparan notas aleatorias en los rangos establecidos
    registra las notas en el archivo de log
    """
    # generate note and send osc message
    note_val = random.randint(0,6)
    synthnames = ['A','B', 'C', 'D', 'E', 'F', 'G']
    ruta = '/scout/note/'+arg_name+'/' + synthnames[note_val]
    ruta = ruta.encode()
    osc_c.send_message(ruta, [1])
    # log to record file
    timetag = time.strftime("%Y%m%d_%H%M%S")
    record_file = open(arg_recfile, 'a+')
    record_file.write("\n[scout.note]: <{0}> {1}\n".format(timetag, ruta.decode()))
    record_file.close()
    return




# -main
def main():
    # -parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', help='Path of capture folder',  default="/media/pi/DATA/capture/video/")
    parser.add_argument('--recfile', help='Path of capture folder',  default="/media/pi/DATA/capture/record/")
    parser.add_argument('--name', help='Name of the directions to scout [NE || SW]',  default="NE")
    parser.add_argument('--verbose', help='Show additional info for debugging',  default=False)
    parser.add_argument('--show', help='Show video',  default=False)
    parser.add_argument('--ip', help='OSC ip',  default="192.168.1.207")
    parser.add_argument('--port', help='OSC port',  default="57120")
    args = parser.parse_args()
    # -init osc client
    osc_addr = args.ip
    osc_port = int(args.port)
    osc_client = OSCClient(osc_addr, osc_port)
    # -load model and labels for detection
    default_model_dir = '/home/pi/Dev/animals/train'
    default_model = 'animals_duo_model.tflite'
    default_labels = 'animals_duo_model.txt'
    args_model = os.path.join(default_model_dir, default_model)
    args_labels = os.path.join(default_model_dir, default_labels)
    args_top_k = 1
    args_camera_idx = 0
    args_threshold = 0.1
    os.makedirs(args.path, exist_ok=True)
    os.makedirs(args.recfile, exist_ok=True)
    # -create the detection interpreter
    print('Cargando {} con {} categorias de objetos.'.format(args_model, args_labels))
    interpreter = cvtf.make_interpreter(args_model)
    interpreter.allocate_tensors()
    labels = load_labels(args_labels)
    # -record file
    timetag = time.strftime("%Y%m%d_%H%M%S")
    arg_recfile = args.recfile + timetag + ".log"
    record_file = open(arg_recfile, 'w+')
    record_file.write("[scout.record.start]:\t----\t----\t-- <{0}>: \n".format(timetag));
    record_file.close()
    # -create a capture object and connect to cam
    cam = None
    witch = 0
    empty = create_blank(640, 480, rgb_color=(0,0,0))
    buffstream = ''
    # -the loop (hole)
    t0 = time.time()
    t2 = time.time()
    nc_a, nc_b, data_sens = dual_detect(args.name, args.verbose)
    while True:
        # -check sensors,
        if (time.time()-t0 > 1):
            nc_a, nc_b, data_sens = dual_detect(args.name, args.verbose)
            t0 = time.time()
        # -then setup capture device
        if (witch == 0):
            if (nc_a > MIN_MASA_DETEC):
                cam = cv2.VideoCapture(4)
                witch = 1
            elif(nc_b > MIN_MASA_DETEC):
                cam = cv2.VideoCapture(0)
                witch = 2
            else:
                #continue
                time.sleep(1)
                pass
        elif(witch == 1):
            if (nc_a > MIN_MASA_DETEC):
                #continue
                pass
            elif(nc_b > MIN_MASA_DETEC):
                cam.release()
                cam = cv2.VideoCapture(0)
                witch = 2
            else:
                cam.release()
                witch = 0
        elif(witch == 2):
            if (nc_a > MIN_MASA_DETEC):
                cam.release()
                cam = cv2.VideoCapture(4)
                witch = 1
            elif(nc_b > MIN_MASA_DETEC):
                #continue
                pass
            else:
                cam.release()
                witch = 0
        # luego, cuando haya un dispositivo activo
        if (witch > 0):
            if (cam.isOpened()):
                # read and convert
                ret, frame = cam.read()
                if not ret:
                    print("-.-* No Video Source")
                    break
                cv2_im = frame
                cv2_im_rgb = cv2.cvtColor(cv2_im, cv2.COLOR_BGR2RGB)
                pil_im = Image.fromarray(cv2_im_rgb)
                # make the classification
                cvtf.set_input(interpreter, pil_im)
                interpreter.invoke()
                results = get_output(interpreter, args_top_k, args_threshold)
                # parse and print results, compare, count!
                # cv2_im = append_results_to_img(cv2_im, results, labels)
                label = labels[results[0][0]]
                percent = int(100 * results[0].score)
                tag = '{}% {}'.format(percent, label)
                ch = ''
                if (label=='Jaguar'): ch='J'
                elif(label=='MexicanGrayWolf'): ch='w'
                elif(label=='Human'): ch='H'
                else: ch = ' '
                # update the buffstream
                buffstream += ch
                if (len(buffstream) > 20): 
                    buffstream = buffstream[1:]
                    if (args.verbose == True):
                        print(buffstream+'/n')
                # count and trigger events, reset buff
                c_J = buffstream.count('J')
                c_W = buffstream.count('w')
                c_H = len(list(filter(lambda x: x == 'H', buffstream)))
                if (c_J>15):
                    lab = "JAGUAR"
                    print("\n\n[->] {0}\n".format(lab))
                    label_callback(lab, witch, args.path, args.name, arg_recfile, data_sens)
                    buffstream = ''
                if (c_W>15):
                    lab = "MexGrayWOLF"
                    print("\n\n[->] {0}\n".format(lab))
                    label_callback(lab, witch, args.path, args.name, arg_recfile, data_sens)
                    buffstream = ''
                if (c_H>15):
                    print("\n\n[->]  .. t[._.]H\n")
                    human_callback(witch, args.path, args.name, arg_recfile, data_sens)
                    buffstream = ''
                # draw image
                if (args.show==True): cv2.imshow('frame', cv2_im)
        else:
            if (args.show==True): cv2.imshow('frame', empty)
            # pass
        # actualiza la maquina de sonido
        if (time.time() - t2 > 30):
            update_soundsystem(arg_recfile, args.name, osc_client)
            t2 = time.time()
        # - detect break key
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()




# ----
if __name__ == '__main__':
    main()