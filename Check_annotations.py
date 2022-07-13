import requests
import numpy as np
import cv2 as cv
import ffmpeg
from ffmpeg import Error
import os
import pandas as pd
import json


# import matplotlib.pyplot as plt


def read_json_dataroom(url):
    r = requests.get(url)

    return r.json()


def find_compets_with_data():
    courses_annotees = []

    base_url = "https://dataroom.liris.cnrs.fr/vizvid_json/pipeline-tracking/"
    # compet = "2021_CF_Montpellier"

    data = read_json_dataroom(base_url)
    compets = [d["name"] for d in data if d['type'] == "directory" and d["name"][:2] == "20"]

    for compet in compets:
        # print(f"Récupération des données dans {compet}")
        compet_url = base_url + compet + '/'
        compet_data = read_json_dataroom(compet_url)
        courses = [d["name"] for d in compet_data if d['type'] == "directory"]
        for course in courses:
            course_url = compet_url + course + '/'
            course_data = read_json_dataroom(course_url)
            for element in course_data:
                if element["name"] == course + '_Espadon.csv':
                    courses_annotees.append(course_url)
                    break
    return (courses_annotees)


def timestr2float(string):
    m, s = string.split(':')
    return (float(m) * 60 + float(s))


def generate_from_above(course_url, size=(1024, 512), destination='./Tracking_Data/from_above'):
    course_name = course_url.split('/')[-2]
    course_json = read_json_dataroom(course_url + course_name + '.json')

    # récupération noms videos droite et gauche
    # Quelquesoit la position de time_offset dans le json ce sera toujours TbuzzerG - TbuzzerD
    for vid_json in course_json['videos']:
        if 'fixeDroite' in vid_json['name']:

            video_droite_url = course_url + vid_json['name']
            destPtsD = np.array(vid_json['destPts']) * np.array(size) / np.array([900, 361])
            srcPtsD = np.array(vid_json['srcPts'])
            HD, _ = cv.findHomography(np.array(srcPtsD), np.array(destPtsD))
            if 'start_flash' in vid_json:
                startD = vid_json['start_flash']
            elif 'start_synchro_flash' in vid_json:
                startD = vid_json['start_synchro_flash']
            elif 'start_momnet' in vid_json:
                startD = vid_json['start_moment']

        elif 'fixeGauche' in vid_json['name']:

            video_gauche_url = course_url + vid_json['name']
            destPtsG = np.array(vid_json['destPts']) * np.array(size) / np.array([900, 361])
            srcPtsG = np.array(vid_json['srcPts'])
            HG, _ = cv.findHomography(np.array(srcPtsG), np.array(destPtsG))
            if 'start_flash' in vid_json:
                startG = vid_json['start_flash']
            elif 'start_synchro_flash' in vid_json:
                startG = vid_json['start_synchro_flash']
            elif 'start_momnet' in vid_json:
                startG = vid_json['start_moment']

    # génération du from_above
    # initialisation video sortie
    fourcc = cv.VideoWriter_fourcc('M', 'J', 'P', 'G')
    out = cv.VideoWriter(os.path.join(destination, course_name + '_from_above.mp4'), fourcc, 50, size)
    print('Dowloading...')
    capG = cv.VideoCapture(video_gauche_url)
    capD = cv.VideoCapture(video_droite_url)

    print('Editing...')

    # commencer au début de chaque vidéo
    _ = capG.set(cv.CAP_PROP_POS_FRAMES, int(np.abs(startG) * 50))
    _ = capD.set(cv.CAP_PROP_POS_FRAMES, int(np.abs(startD) * 50))

    while capG.isOpened() and capD.isOpened():
        retG, frameG = capG.read()
        retD, frameD = capD.read()

        if not (retG and retD):
            print('Done!')
            print('Fichier sortie:', os.path.join(destination, course_name + '_from_above.mp4'))
            break

        imgD = cv.warpPerspective(frameD, HD, size)
        imgG = cv.warpPerspective(frameG, HG, size)
        img_above = np.where(imgD != 0, imgD, imgG)

        out.write(img_above)

    capG.release()
    capD.release()
    out.release()


def generer_data(course_url):
    course_name = course_url.split('/')[-2]

    print('Génération du from_above...')
    generate_from_above(course_url)

    print('Découpage des lignes de course...')
    course_Espadon = pd.read_csv(course_url + course_name + '_Espadon.csv')
    swimmerId = np.array(list(course_Espadon['swimmerId'].unique()))
    lignes = swimmerId + 1
    course_json = read_json_dataroom(course_url + course_name + '.json')
    decoupe_video('./Tracking_Data/from_above/' + course_name + '_from_above.mp4', course_json, course_Espadon, lignes,
                  nbl=8, course_name=course_name, destination='./Tracking_Data/Lignes/')

    print('Interolation des données Espadon...')
    for swimId in swimmerId:  # interpolation
        dt = course_Espadon.loc[course_Espadon['swimmerId'] == swimId].sort_values('frameId')
        frames_connues = np.array(dt['frameId'])
        events_connus = np.array(dt['eventX'])
        interpolation = np.empty(frames_connues[-1])
        interpolation[:] = np.nan
        interpolation[frames_connues - 1] = events_connus
        interpolation[0] = 0
        dtt = pd.DataFrame(interpolation)
        dtt.interpolate(inplace=True)
        interpole = np.array(dtt)[:, 0]
        ligne = swimId + 1
        np.save(os.path.join('./Tracking_Data/Positions/', course_name + "ligne_" + str(ligne) + "_pos.npy"), interpole)


def decoupe_video(file_in, course_json, course_Espadon, lignes, nbl=8, course_name="",
                  destination='./Tracking_Data/Lignes/', only_annoted=True):  # nbl: number of lines

    video_data = ffmpeg.probe(file_in)
    h = video_data['streams'][0]['height']
    w = video_data['streams'][0]['width']

    for ligne in lignes:
        if ligne >= 1 and ligne <= 8:
            position_x = str(0)
            position_y = str(max(0, int((ligne - 1) / nbl * h)))
            largeur = str(w)
            hauteur = str(min(h / nbl, h - int(position_y)))

            if only_annoted:
                stop_time = course_Espadon.loc[course_Espadon['swimmerId'] == ligne - 1]['frameId'].max() / 50
            else:
                stop_time = timestr2float(course_json['temps']["temps" + str(ligne)])

            (
                # va cherher la vidéo et decoupe comme on souhaite, le crop est écrit dans ligne_{ligne}.m4v dans le dossier destination
                ffmpeg
                .input(file_in, to=str(0.01 + stop_time))  # on s'arrete à la fin de la course
                .filter("crop", largeur, hauteur, position_x, position_y)
                .output(os.path.join(destination, course_name + "ligne_" + str(ligne) + ".m4v"))
                .global_args("-loglevel", "error")
                .overwrite_output()
                .run()
            )
        else:
            pass
            # print(f"Ligne {ligne}, invalide. ligne doit être un entier entre 1 et 8")


def visu_annotation(course, frameid=[], eventx=[], fps=50, save=False):
    rayon_int = 3
    largeur_cercle_ext = 2
    color_int = np.array([0, 51, 255])
    color_ext = np.array([51, 51, 153])

    i, j = np.meshgrid(np.arange(64), np.arange(1024), indexing='ij')

    if save:
        # initialisation video sortie
        fourcc = cv.VideoWriter_fourcc('M', 'J', 'P', 'G')
        out = cv.VideoWriter(os.path.join('Visu.mp4'), fourcc, fps, (1024, 64))
    cap = cv.VideoCapture(course)
    count = 0
    while cap.isOpened():
        count += 1
        ret, frame = cap.read()

        if not ret:
            print('End of stream')
            break

        # cv.waitKey(1000//25)
        if cv.waitKey(1000 // fps) == ord('q'):  # quitter en appuyant sur q
            break

        for ind, framei in enumerate(frameid):
            if 0 < framei - count <= 100:

                temp = (i - 32) ** 2 + (j - (1024 - eventx[ind] / 50 * 1024)) ** 2

                disque_int = (temp < rayon_int ** 2)
                rayon_ext = rayon_int + largeur_cercle_ext + 0.16 * max(0, min(100, framei - count))
                cercle_ext = (temp <= rayon_ext ** 2)
                if rayon_ext > largeur_cercle_ext:
                    cercle_ext = ((rayon_ext - largeur_cercle_ext) ** 2 <= temp) * cercle_ext
                frame[cercle_ext] = color_ext
                frame[disque_int] = color_int

        cv.imshow('Course', frame)
        if save:
            out.write(frame)
    cap.release()
    if save:
        out.release


def visu_data(course, pos_link="", fps=50, save=False):
    largeur = 2
    color = np.array([0, 51, 255])
    posX = np.load(pos_link)
    i, j = np.meshgrid(np.arange(64), np.arange(1024), indexing='ij')

    if save:
        # initialisation video sortie
        fourcc = cv.VideoWriter_fourcc('M', 'J', 'P', 'G')
        out = cv.VideoWriter(os.path.join('Visu.mp4'), fourcc, fps, (1024, 64))
    cap = cv.VideoCapture(course)
    ind = 0
    while cap.isOpened():

        ret, frame = cap.read()

        if not ret:
            print('End of stream')
            break

        # cv.waitKey(1000//25)
        if cv.waitKey(1000 // fps) == ord('q'):  # quitter en appuyant sur q
            break

        posInPool = 1024 * (1 - posX[min(ind, len(posX) - 1)] / 50)

        bar = (j <= posInPool + largeur / 2) * (j >= posInPool - largeur / 2)
        frame[bar] = color

        cv.imshow('Course', frame)
        if save:
            out.write(frame)
        ind += 1
    cap.release()
    if save:
        out.release


def demi_tour_impossible(eventX):
    for i in range(1, len(eventX) - 1):
        if ((eventX[i - 1] - eventX[i]) * (eventX[i] - eventX[i + 1]) < 0 and eventX[i] < 49.5 and eventX[i] > 0.5):
            return (True)
    return (False)


if __name__ == "__main__":

    print('Détection des courses...')
    courses_annotees = []
    courses_non_annotees = []

    base_url = "https://dataroom.liris.cnrs.fr/vizvid_json/pipeline-tracking/"
    # compet = "2021_CF_Montpellier"

    data = read_json_dataroom(base_url)
    compets = [d["name"] for d in data if d['type'] == "directory" and d["name"][:2] == "20"]

    for compet in compets:
        # print(f"Récupération des données dans {compet}")
        compet_url = base_url + compet + '/'
        compet_data = read_json_dataroom(compet_url)
        courses = [d["name"] for d in compet_data if d['type'] == "directory"]
        for course in courses:
            course_url = compet_url + course + '/'
            course_name = course_url.split('/')[-2]
            if course_name[0] == "2":
                course_data = read_json_dataroom(course_url)
                annote = False
                for element in course_data:
                    if element["name"] == course + '_Espadon.csv':
                        courses_annotees.append(course_url)
                        annote = True
                        break
                if not (annote):
                    courses_non_annotees.append(course_url)

    print("Génération du json...")
    rep = []

    for course_url in courses_annotees:
        course_name = course_url.split('/')[-2]
        course_Espadon = pd.read_csv(course_url + course_name + '_Espadon.csv')
        course_json = read_json_dataroom(course_url + course_name + '.json')
        swimmerId = np.array(list(course_Espadon['swimmerId'].unique()))
        temp = np.zeros(8)
        for i in range(8):
            dico = {}
            dico['file_csv'] = course_name + '_Espadon.csv'
            dico["nameCompetition"] = course_url.split('/')[-3]
            dico["nameCourse"] = course_name
            dico["line"] = i + 1
            if i in swimmerId:
                swimId = i
                dt = course_Espadon.loc[course_Espadon['swimmerId'] == swimId].sort_values('frameId')
                eventX = np.array(dt['eventX'])
                pb_demi_tour = demi_tour_impossible(eventX)
                derniere_annotation = np.array(dt['frameId']).max() / 50
                dico['incoherentTurn'] = int(pb_demi_tour)
                try:
                    temps_fin = timestr2float(course_json['temps']["temps" + str(swimId + 1)])
                    incomplete = (np.abs(derniere_annotation - temps_fin) > 3)
                    dico['race_data'] = 'PARTIAL' if incomplete else 'COMPLETE'
                except:
                    dico['race_data'] = 'UNKNOWN'
            else:
                dico['race_data'] = 'EMPTY'
            rep.append(dico)

    for course_url in courses_non_annotees:
        for i in range(8):
            dico = {}
            course_name = course_url.split('/')[-2]
            dico["nameCompetition"] = course_url.split('/')[-3]
            dico["nameCourse"] = course_name
            dico["line"] = i + 1
            dico['race_data'] = 'EMPTY'
            rep.append(dico)

    data = json.dumps(rep)
    with open('qualite_espadon.json', 'w') as outfile:
        outfile.write(data)
