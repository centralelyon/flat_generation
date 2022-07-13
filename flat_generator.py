import os
import json
# from urllib.request import urlopen
# import ujson
# from MediaInfo import MediaInfo
# import yaml
import ffmpeg
import pandas as pd
import numpy as np

from Check_annotations import demi_tour_impossible, timestr2float
from utils import upload_json_dataroom

cmd_ffprobe = "/opt/homebrew/bin/ffprobe"
path_theo = "/home/liris/vizvid_data/files/admin/files/pipeline-tracking/"
path_romain = "/Users/rvuillem/Desktop/dataroom/pipeline-tracking/"
path_romain_small = "/Users/rvuillem/Desktop/pipeline-tracking-small/"
path_clement = "./pipeline/"
path_dataroom = "/home/liris/vizvid_data/files/admin/files/pipeline_tracking/"

path = path_dataroom


def dicotiser(f):
    with open(f, encoding='utf-8') as json_data:
        return (json.load(json_data))


def istherejson(run_files):
    for file in run_files:
        if file.split('.')[-1] == "json":
            return (True)
    return (False)


def isthereespadon(run_files):
    for file in run_files:
        if "Espadon" in file.split('.')[0].split('_'):
            return (True)
    return (False)


def isthereespadonmodifie(run_files):
    for file in run_files:
        if "_Espadon_edited" in file.split('.')[0]:
            return (True)
    return (False)


def isthereautomatic(run_files):
    for file in run_files:
        if "_automatique" in file.split('.')[0]:
            return (True)
    return (False)


def checkRun(run_link, res, previous_data, flat_old):
    run_name = run_link.split('/')[-1]
    run_name_splited = run_name.split('_')
    if len(run_name_splited) >= 6:
        run_files = os.listdir(run_link)
        # computing general data of the run
        common_data = {}
        common_data["distance"] = run_name_splited[4]
        common_data["json"] = istherejson(run_files)
        common_data["nameCompetition"] = run_link.split('/')[-2]
        common_data["nameCourse"] = run_name
        common_data["sexe"] = run_name_splited[3]
        common_data["epreuve"] = run_name_splited[2]
        common_data['espadon'] = isthereespadon(run_files)
        common_data["espadonModifie"] = isthereespadonmodifie(run_files)
        common_data["automatic"] = isthereautomatic(run_files)

        for file in run_files:
            if file.lower().endswith(('.mp4', '.kmv', '.avi', 'mov')):
                video_dic = common_data.copy()
                video_dic['name'] = file
                video_dic['conteneur'] = file.split('.')[-1]
                video_dic["type_video"] = file.split('.')[0].split('_')[-1]

                # video data ( ffmpeg.probe)
                probe = ffmpeg.probe(run_link + '/' + file)
                video_dic["fileSize"] = probe['format']['size']

                # video stream data
                for stream in probe['streams']:  # find video stream
                    if stream['codec_type'] == 'video':
                        video_stream = stream
                        break
                video_dic['duration'] = float(video_stream['duration'])
                video_dic["videoWidth"] = int(video_stream['width'])
                video_dic["videoHeight"] = int(video_stream['height'])
                video_dic["videoAspectRatio"] = float(video_dic["videoWidth"] / video_dic["videoHeight"])
                video_dic["videoCodec"] = video_stream['codec_tag_string']
                video_dic["videoFrameRate"] = int(video_stream['r_frame_rate'].split('/')[0])
                video_dic["videoFrameCount"] = int(video_stream['nb_frames'])

                # json data
                if common_data["json"]:
                    run_json = dicotiser(run_link + '/' + run_name + '.json')
                    if "one_is_up" in run_json:
                        video_dic["one_is_up"] = run_json["one_is_up"]
                    if "start_moment" in run_json:
                        video_dic["start_moment"] = run_json["start_moment"]
                    if "start_moment" in run_json:
                        video_dic["start_moment"] = run_json["start_moment"]
                    if "start_side" in run_json:
                        video_dic["start_side"] = run_json["start_side"]
                    if "flash" in run_json:
                        video_dic["flash"] = run_json["flash"]
                    if "start_flash" in run_json:
                        video_dic["start_flash"] = run_json["start_flash"]

                # annotation data
                thereisdata = common_data["espadonModifie"] or common_data["espadon"]
                if common_data["espadonModifie"]:
                    run_annotation = pd.read_csv(run_link + '/' + run_name + '_Espadon_edited.csv')
                elif common_data["espadon"]:
                    run_annotation = pd.read_csv(run_link + '/' + run_name + '_Espadon.csv')

                video_dic["completeness"] = 0.0
                if thereisdata:
                    swimmerId = list(run_annotation['swimmerId'].unique())
                    c = 0
                    video_dic['time_available'] = False
                    for swimId in swimmerId:
                        dt = run_annotation.loc[run_annotation['swimmerId'] == swimId].sort_values('frameId')
                        eventX = np.array(dt['eventX'])
                        if demi_tour_impossible(eventX):
                            video_dic["incoherent_turn"] = True
                        last_annotation = np.array(dt['frameId']).max() / video_dic["videoFrameRate"]
                        time = run_json['temps']["temps" + str(swimId + 1)]
                        if time != "ATAbsence de temps" and time != None:
                            stop_time = timestr2float(time)
                            video_dic["completeness"] += (stop_time - last_annotation) / stop_time
                            c += 1
                        else:
                            video_dic['time_available'] = False
                    if c != 0.0:
                        video_dic["completeness"] /= c

                # gestion de data_checked
                if previous_data:
                    b = True
                    for old_dic in flat_old:
                        if old_dic["name"] == video_dic["name"]:
                            video_dic['data_checked'] = old_dic['data_checked']
                            b = False
                            break
                    if b:
                        video_dic["data_checked"] = False

                else:
                    video_dic["data_checked"] = False
                res.append(video_dic)


def main():
    competitions = next(os.walk(path))[1]
    res = []
    previous_data = False
    if "Flat.json" in os.listdir(path):
        previous_data = True
        flat_old = dicotiser(path + 'Flat.json')
    else:
        flat_old = []
    for compet in competitions:
        if compet[0:2] == "20":  # eclude non competition folders
            runs = next(os.walk(path + compet))[1]
            for run in runs:
                checkRun(path + compet + "/" + run, res, previous_data, flat_old)
    return res


if __name__ == '__main__':
    flat = main()

    upload_json_dataroom(flat, "pipeline-tracking/flat.json")
    # with open('Flat_test.json', 'w', encoding='utf-8') as f:
    #     json.dump(flat, f, ensure_ascii=False, indent=4)

    """
    with open("config-neptune-dev.yml", "r") as stream:

        try:
            config = yaml.safe_load(stream) # for config["PIPELINE-TRACKING"]

            dic, flat = main_online()

            with open("../meta.json", 'w') as wjson:
                json.dump(dic, wjson, ensure_ascii=False, sort_keys=True, indent=4)

            with open("../flat.json", 'w') as wjson:
                json.dump(flat, wjson, ensure_ascii=False, sort_keys=True, indent=4)

        except yaml.YAMLError as exc:
            print("Yaml error", exc)
    """

    """
        flat = {}
        flat["container"] = "string; conteneur de la vidéo"
        flat["distance"] = 'int; longeur de la course'
        flat['duration'] = 'float; durée de la vidéo en secondes'
        flat['fileSize'] = "int; taille du fichier en octets"
        flat['json'] = "boolean; si un json est disponible pourla course"
        flat['name'] = "string; nom du fichier video"
        flat['nameCompetition'] = "string; nom de la compétition"
        flat["nameCourse"] = "string; nom de la course"
        flat['one_is_up'] = 'boolean; si la ligne numéro 1 est en haut de l\'image'
        flat['sexe'] = 'string; hommes ou dames'
        flat['start_moment'] = 'float; temps départ de la course en seconde'
        flat['start_side'] = "boolean; si le départ est visible dans la vidéo"
        flat['type_video'] = 'string; fixeDroite ou fixeGauche ou positions_auto ou vueDessus ou ...'
        flat["videoAspectRatio"] = "float; longueur/largeur des frames de la vidéo"
        flat["videoCodec"] = "string; codec utilisé pour compresser la vidéo"
        flat["videoFrameCount"] = "int; nombre de frames dans la vidéo"
        flat["videoFrameRate"] ="float; fps de la vidéo"
        flat["videoHeight"] = "int; hauteur des frames de la vidéo"
        flat["videoWidth"] = "int; largeur de la vidéo"
        flat['epreuve'] = "string; type d'epreuve (à définir) "
        flat['espadon'] = 'boolean; si les données espadons de la course sont disponibles dans la pipeline'
        flat['espadonModifie'] = 'boolean; si des données espadons modifiées sont disponibles dans la pipeline'
        flat['automatic'] = 'boolean; si des données de tracking automatique sont disponibles dans la pipeline'
        flat['data_checked'] = 'boolean; si un humain à vérifier lesq données à la main'
        flat['incoherent_turn'] = 'boolean; si les données annotées présente un \"demi-tour\" incohérent'
        flat['completeness'] = 'float; proportion des données annotées'
        flat['flash'] = 'list of tuples of ints; fenetre du flash'
        flat['start_flash'] ='float; sépart calculé par flash'
        flat['time_available'] = "boolean; si le temps du nageur est disponible pour tous les nageurs (si faux, completeness n'est pas fiable)"

        with open('Flat_description.json', 'w', encoding='utf-8') as f:
            json.dump(flat, f, ensure_ascii=False, indent=4)
            
    """
