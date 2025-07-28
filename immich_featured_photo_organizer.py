import sys
import argparse
import logging
import requests
import time
import datetime
import numpy as np
import cv2
from math import exp
import random

requests_kwargs = {}
current_datetime = datetime.datetime.now(datetime.timezone.utc)
people_data_cache = {}
updated_people_names = []
blur_score_cache = {}
asset_count = 0


def main():
    global requests_kwargs, asset_count
    start_time = time.time()
    args = parseargs()
    root_url = args["api_url"]
    api_key = args["api_key"]
    fetch_chunk_size = args["fetch_chunk_size"]
    unattended = args["unattended"]
    specific_people_ids = args["people_ids"]
    specific_people_names = args["people_names"]
    favorite_people = args["favorite_people"]
    random_mode = args["random_mode"]
    low_pixel_mode = args["low_pixel_mode"]
    detect_blur = args["detect_blur"]
    detect_blur_reversed = args["detect_blur_reversed"]
    recency_bias = args["recency_bias"]
    recency_bias_reversed = args["recency_bias_reversed"]
    date_bias = args["date_bias"]
    birthday_bias = args["birthday_bias"]
    sigma_days = args["sigma_days"]

    logging.debug("root_url = %s", root_url)
    logging.debug("api_key = %s", api_key)
    logging.debug("fetch_chunk_size = %d", fetch_chunk_size)
    logging.debug("unattended = %s", unattended)
    logging.debug("specific_people_ids = %s", specific_people_ids)
    logging.debug("specific_people_names = %s", specific_people_names)
    logging.debug("favorite_people = %s", favorite_people)
    logging.debug("random_mode = %s", random_mode)
    logging.debug("low_pixel_mode = %s", low_pixel_mode)
    logging.debug("detect_blur = %s", detect_blur)
    logging.debug("detect_blur_reversed = %s", detect_blur_reversed)
    logging.debug("recency_bias = %s", recency_bias)
    logging.debug("recency_bias_reversed = %s", recency_bias_reversed)
    logging.debug("date_bias = %s", date_bias)
    logging.debug("birthday_bias = %s", birthday_bias)
    logging.debug("sigma_days = %s", sigma_days)

    # Request arguments for API calls
    requests_kwargs = {
        'headers': {
            'x-api-key': api_key,
            'Content-Type': 'application/json',
            'Accept': 'application/json, image/*'
        }
    }

    # append trailing slash and 'api' to root URL
    if root_url[-1] != '/':
        if root_url[-3:] != 'api':
            root_url += '/api/'
        else:
            root_url += '/'
    elif root_url[-4:] != 'api/':
        root_url += 'api/'

    server_version = fetch_server_version(root_url)
    if server_version['major'] == 1 and server_version['minor'] <= 105:
        logging.error("This code only works for version >1.105")
        exit(1)

    if favorite_people and server_version['major'] == 1 and server_version['minor'] < 126:
        logging.error("The '--favorite-people' option only works for version >=1.126")
        exit(1)

    logging.info("Requesting all people")
    people_data = fetch_people_data(root_url, person_ids=specific_people_ids, person_names=specific_people_names)
    logging.info("%d named people found", len(people_data))

    if people_data == [{}]:
        logging.error("Cannot proceed with 0 people found.")
        exit(1)

    logging.info("Sorting people that are named...")
    people_with_names = filter_people_data(people_data, birthday_bias, favorite_people)

    extra_info = ""
    if birthday_bias and favorite_people:
        extra_info = "with birthdays and being favorite"
    elif birthday_bias and not favorite_people:
        extra_info = "with birthdays"
    elif favorite_people:
        extra_info = "being favorite"
    logging.info("%d people identified %s", len(people_with_names), extra_info)

    if len(people_with_names) == 0:
        logging.error("Cannot proceed with 0 people identified.")
        exit(1)

    if not unattended:
        print("Press Enter to continue, Ctrl+C to abort")
        input()

    person_scores = []
    for person in people_with_names:
        person_score = {}
        person_score['person_id'] = person['id']
        person_score['person_name'] = person['name']
        person_score['person_birthDate'] = person['birthDate']
        person_score['assets'] = []
        person_assets = fetch_assets_with_people(root_url, [person_score['person_id']],
                                                 fetch_chunk_size=fetch_chunk_size)

        if not person_assets:
            logging.warning("Could not find any assets for person id: '%s' name: '%s'", person_score['person_id'],
                            person_score['person_name'])
            continue

        if len(person_assets) == 1:
            person_score['assets'].append({'asset_id': person_assets[0], 'asset_score': 1})
            person_scores.append(person_score)
            continue

        oldest_asset_datetime = ''
        if birthday_bias and recency_bias_reversed:
            oldest_asset_datetime = find_oldest_datetime(person_assets)

        for asset in person_assets:
            if asset['visibility'] != 'timeline' or asset['isTrashed']:
                continue

            asset_score = 1.0
            asset_data = {'asset_id': asset['id']}
            asset_datetime = asset['exifInfo']['dateTimeOriginal']
            if random_mode:
                if birthday_bias and not is_date_asset(asset_datetime, person_score['person_birthDate'],
                                                       proximity_days=sigma_days):
                    asset_data = {}
                if date_bias and not is_date_asset(asset_datetime, date_bias, proximity_days=sigma_days):
                    asset_data = {}

            elif not random_mode:
                face_size = calculate_face_size(asset, person_score['person_id'])

                if low_pixel_mode:
                    face_size = 1.0 / face_size

                blur_score = 1.0
                if detect_blur or detect_blur_reversed:
                    blur_score = calculate_blur(root_url, asset['id'], reversed_blur=detect_blur_reversed)

                date_score = 1.0
                if birthday_bias:
                    date_score = calculate_birthday_bias(asset_datetime,
                                                         person_score['person_birthDate'],
                                                         oldest_datetime_str=oldest_asset_datetime, sigma=sigma_days,
                                                         is_recency=recency_bias,
                                                         is_recency_reversed=recency_bias_reversed,
                                                         biased_date=date_bias)

                elif recency_bias or recency_bias_reversed:
                    date_score = calculate_datetime_recency(asset_datetime,
                                                            curr_datetime=current_datetime,
                                                            reversed_recency=recency_bias_reversed)
                elif date_bias:
                    date_score = calculate_date_bias(asset_datetime, date_bias, sigma=sigma_days)

                asset_score = face_size * blur_score * date_score
                logging.verbose("Score for asset_id %s of person_id %s and person_name %s: %d with face_size=%d, "
                              "blur_score=%d, date_score=%d", asset_data['asset_id'], person_score['person_id'],
                              person_score['person_name'], asset_score, face_size, blur_score, date_score)

            if not random_mode:
                asset_data['asset_score'] = asset_score

            if asset_data:
                person_score['assets'].append(asset_data)
                asset_count += 1

        person_scores.append(person_score)

    for person_score in person_scores:
        if not person_score['assets']:
            logging.warning("Couldn't find any assets for person_id: %s person_name: %s", person_score['person_id'],
                            person_score['person_name'])
            continue

        featured_asset_id = find_featured_photo(person_score['assets'], is_random=random_mode)

        if len(featured_asset_id) == 0:
            logging.warning("Couldn't find an suitable asset for person_id: %s person_name: %s",
                            person_score['person_id'], person_score['person_name'])
            continue

        update_person_featured(root_url, person_score['person_id'], person_score['person_name'], featured_asset_id)

    logging.info("Done!")
    logging.info("Updated the following people: %s", ', '.join(updated_people_names))
    logging.debug("--- Execution took %s seconds ---" % (time.time() - start_time))
    logging.debug("Assets processed: %d", asset_count)


def parseargs():
    parser = argparse.ArgumentParser(
        description="Automatically select the best featured foto for a person in Immich",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("api_url", help="The root API URL of Immich, e.g. https://immich.mydomain.com/api/")
    parser.add_argument("api_key", help="The Immich API Key to use")
    parser.add_argument("-u", "--unattended", action="store_true",
                        help="Do not ask for user confirmation after identifying people. "
                             "Set this flag to run script as a cronjob.")
    parser.add_argument("-s", "--people-ids", action="extend", type=str, nargs="*", default=[],
                        help="Select featured photos for specific people using ids")
    parser.add_argument("-S", "--people-names", action="extend", type=str, nargs='*', default=[],
                        help="Select featured photos for specific people using names")
    parser.add_argument("-f", "--favorite-people", action="store_true",
                        help="Select featured photos for favorite people")
    parser.add_argument("-r", "--random-mode", action="store_true",
                        help="Select featured photos randomly")
    parser.add_argument("-p", "--low-pixel-mode", action="store_true",
                        help="Favor low resolution photos")
    parser.add_argument("-b", "--detect-blur", action="store_true",
                        help="Detect if images are blurry, favoring sharpness")
    parser.add_argument("-B", "--detect-blur-reversed", action="store_true",
                        help="Detect if images are blurry, favoring blurriness")
    parser.add_argument("-t", "--recency-bias", action="store_true",
                        help="Take image recency into account, newest to oldest")
    parser.add_argument("-T", "--recency-bias-reversed", action="store_true",
                        help="Take image recency into account, oldest to newest")
    parser.add_argument("-d", "--date-bias", type=str,
                        help="Favor pictures taken around a certain date: 'yyyy-mm-dd' or 'mm-dd' or 'yyyy'")
    parser.add_argument("-n", "--birthday-bias", action="store_true",
                        help="Favor photod around a person's birthday. Can be combined with -t and -T.")
    parser.add_argument("-g", "--sigma-days", type=positive_float, default=1.0,
                        help="Set the sigma days when using -d, -D or -n")
    parser.add_argument("-c", "--fetch-chunk-size", default=1000, type=int,
                        help="Maximum number of assets and people to fetch with a single API call")
    parser.add_argument("-l", "--log-level", default="INFO",
                        choices=['CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG', 'VERBOSE'],
                        help="Log level to use")
    args = vars(parser.parse_args())

    add_logging_level('VERBOSE', logging.DEBUG - 5)
    # set up logger to log in logfmt format
    logging.basicConfig(level=args["log_level"], stream=sys.stdout,
                        format='time=%(asctime)s level=%(levelname)s msg=%(message)s')
    logging.Formatter.formatTime = (lambda self, record, datefmt=None: datetime.datetime.fromtimestamp(record.created,
                                                                                                       datetime.timezone.utc).astimezone().isoformat(
        sep="T", timespec="milliseconds"))

    return args


def positive_float(val):
    try:
        fval = float(val)
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid float value: '{val}'")
    if fval <= 0:
        raise argparse.ArgumentTypeError("Value must be a positive float (> 0)")
    return fval


def add_logging_level(levelName, levelNum, methodName=None):
    """
    Comprehensively adds a new logging level to the `logging` module and the
    currently configured logging class.

    `levelName` becomes an attribute of the `logging` module with the value
    `levelNum`. `methodName` becomes a convenience method for both `logging`
    itself and the class returned by `logging.getLoggerClass()` (usually just
    `logging.Logger`). If `methodName` is not specified, `levelName.lower()` is
    used.

    To avoid accidental clobberings of existing attributes, this method will
    raise an `AttributeError` if the level name is already an attribute of the
    `logging` module or if the method name is already present

    Example
    -------
    >>> add_logging_level('TRACE', logging.DEBUG - 5)
    >>> logging.getLogger(__name__).setLevel("TRACE")
    >>> logging.getLogger(__name__).trace('that worked')
    >>> logging.trace('so did this')
    >>> logging.TRACE
    5

    """
    if not methodName:
        methodName = levelName.lower()

    if hasattr(logging, levelName):
        raise AttributeError('{} already defined in logging module'.format(levelName))
    if hasattr(logging, methodName):
        raise AttributeError('{} already defined in logging module'.format(methodName))
    if hasattr(logging.getLoggerClass(), methodName):
        raise AttributeError('{} already defined in logger class'.format(methodName))

    # This method was inspired by the answers to Stack Overflow post
    # http://stackoverflow.com/q/2183233/2988730, especially
    # http://stackoverflow.com/a/13638084/2988730
    def log_for_level(self, message, *args, **kwargs):
        if self.isEnabledFor(levelNum):
            self._log(levelNum, message, args, **kwargs)

    def log_to_root(message, *args, **kwargs):
        logging.log(levelNum, message, *args, **kwargs)

    logging.addLevelName(levelNum, levelName)
    setattr(logging, levelName, levelNum)
    setattr(logging.getLoggerClass(), methodName, log_for_level)
    setattr(logging, methodName, log_to_root)


def calculate_face_size(asset_data: dict, p_id: str) -> float:
    asset_person_data = {}

    for dict_data in asset_data['people']:
        if p_id in dict_data.values():
            asset_person_data = dict_data
            break

    if not asset_person_data:
        return 1.0

    # NOTE: you can't specify which face [of the same person] is to be used for
    # the featured photo in the `/people` API call currently. Thus it only makes
    # sense to check the first detected face [of the same person]. This has an impact
    # on screenshots containing multiple photos and collages.
    box_x1 = asset_person_data['faces'][0]['boundingBoxX1']
    box_x2 = asset_person_data['faces'][0]['boundingBoxX2']
    box_y1 = asset_person_data['faces'][0]['boundingBoxY1']
    box_y2 = asset_person_data['faces'][0]['boundingBoxY2']

    # resize factor needs to be calculated because Immich doesn't send the full image to
    # the ML server.
    ml_image_size = asset_person_data['faces'][0]['imageWidth'] * asset_person_data['faces'][0]['imageHeight']
    real_image_size = asset_data['exifInfo']['exifImageWidth'] * asset_data['exifInfo']['exifImageHeight']
    resize_factor = real_image_size / ml_image_size
    return calculate_real_face_size(box_x1, box_x2, box_y1, box_y2, resize_factor)


def calculate_real_face_size(x1: int, x2: int, y1: int, y2: int, r_factor: float) -> float:
    return abs((x1 - x2) * (y1 - y2)) * r_factor


def calculate_blur(root_url: str, asset_id: str, reversed_blur: bool = False) -> float:
    if asset_id not in blur_score_cache:
        image_bytes = fetch_asset_image(root_url, asset_id)
        image = cv2.imdecode(image_bytes, -1)
        if image is None or image.size == 0:
            blur_score_cache[asset_id] = 1.0
        else:
            image = cv2.resize(image, (300, 300), interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            laplacian_variance = cv2.Laplacian(gray, cv2.CV_64F).var()
            if reversed_blur:
                laplacian_variance = 1.0 / laplacian_variance

            blur_score_cache[asset_id] = laplacian_variance

    return blur_score_cache[asset_id]


def calculate_datetime_recency(datetime_str: str, curr_datetime: datetime.datetime = None,
                               reversed_recency: bool = False, decay_rate: float = 0.004) -> float:
    if curr_datetime is None:
        curr_datetime = datetime.datetime.now()

    dt = datetime.datetime.fromisoformat(datetime_str)
    days_ago = (curr_datetime - dt).total_seconds() / 86400
    days_ago = max(days_ago, 0)
    direction = 1 if reversed_recency else -1

    # decay_rate = ln(2) / half life in days
    return exp(direction * decay_rate * days_ago)


def calculate_datetime_proximity(datetime_str: str, target_date_str: str, sigma: float = None) -> float:
    if sigma is None:
        sigma = 1.0
    dt = datetime.datetime.fromisoformat(datetime_str).astimezone(datetime.timezone.utc)
    target_date = datetime.datetime.fromisoformat(target_date_str).astimezone(datetime.timezone.utc)
    delta_days = abs((dt - target_date).total_seconds()) / 86400
    return exp(- (delta_days ** 2) / (2 * sigma ** 2))


def calculate_datetime_proximity_yearless(datetime_str: str, target_mm_dd: str, sigma: float = None) -> float:
    if sigma is None:
        sigma = 1.0
    dt = datetime.datetime.fromisoformat(datetime_str)
    dt_doy = dt.timetuple().tm_yday
    target_mm_dd = datetime_str[:5] + target_mm_dd  # need to account for leap years with doy
    target_date = datetime.datetime.fromisoformat(target_mm_dd)
    target_doy = target_date.timetuple().tm_yday

    # Leap years
    year_length = 366 if dt.year % 4 == 0 and (dt.year % 100 != 0 or dt.year % 400 == 0) else 365

    delta = abs(dt_doy - target_doy)
    circular_delta = min(delta, year_length - delta)
    return exp(- (circular_delta ** 2) / (2 * sigma ** 2))


def calculate_date_bias(asset_dt: str, target_date: str, sigma: float = None) -> float:
    if len(asset_dt) == 0 or len(target_date) == 0:
        return 1.0

    len_date_bias = len(target_date)
    if len_date_bias == 10:
        return calculate_datetime_proximity(asset_dt, target_date, sigma=sigma)
    elif len_date_bias == 5:
        return calculate_datetime_proximity_yearless(asset_dt, target_date, sigma=sigma)
    elif len_date_bias == 4:
        # with a sigma of 183 days and a date_bias of yyyy-07-02, this should catch yyyy-01-01 up until
        # (yyyy+1)-01-01
        date_bias_adjusted = target_date + '-07-02'
        return calculate_datetime_proximity(asset_dt, date_bias_adjusted, sigma=183)
    return 1.0


def find_oldest_datetime(assets: list) -> str:
    return min([asset['exifInfo']['dateTimeOriginal'] for asset in assets])


def get_adjusted_birthday(birthdate_str: str, oldest_datetime_str: str = '', is_recency: bool = False,
                          is_recency_reversed: bool = False, biased_date: str = '') -> str:
    if is_recency:
        birthdate_str = str(current_datetime.year) + birthdate_str[4:]
        birthday_adj = datetime.date.fromisoformat(birthdate_str)
        is_future = (current_datetime.date() - birthday_adj).total_seconds() < 0
        if is_future:
            birthday_adj = birthday_adj.replace(year=current_datetime.year - 1)
        return str(birthday_adj)

    elif is_recency_reversed and len(oldest_datetime_str) > 0:
        oldest_date = datetime.datetime.fromisoformat(oldest_datetime_str).date()
        birthdate_date = datetime.date.fromisoformat(birthdate_str)
        birthday_adj = birthdate_date.replace(year=oldest_date.year)
        is_past = (oldest_date - birthday_adj).total_seconds() > 0
        if is_past:
            birthday_adj = birthday_adj.replace(year=oldest_date.year + 1)
        return str(birthday_adj)
    elif len(biased_date) == 4:
        return biased_date + birthdate_str[4:]
    else:
        birthdate_adj = birthdate_str[5:]
        return str(birthdate_adj)


def calculate_birthday_bias(asset_datetime_str: str, birthdate: str, oldest_datetime_str: str = '',
                            sigma: float = None, is_recency: bool = False, is_recency_reversed: bool = False,
                            biased_date: str = '') -> float:
    # takes recency, yearless or a specific year into account
    birthday_adjusted = get_adjusted_birthday(birthdate,
                                              oldest_datetime_str=oldest_datetime_str, is_recency=is_recency,
                                              is_recency_reversed=is_recency_reversed, biased_date=biased_date)
    len_birthday_adjusted = len(birthday_adjusted)
    if len_birthday_adjusted in [10, 4]:
        return calculate_datetime_proximity(asset_datetime_str, birthday_adjusted, sigma=sigma)
    elif len_birthday_adjusted == 5:
        return calculate_datetime_proximity_yearless(asset_datetime_str, birthday_adjusted, sigma=sigma)
    return 1.0


def is_date_asset(asset_datetime_str: str, target_date_str: str, proximity_days) -> bool:
    if proximity_days is None:
        proximity_days = 1

    if proximity_days == 0:
        return asset_datetime_str[5:10] == target_date_str[5:]

    len_target_date_str = len(target_date_str)
    if len_target_date_str == 4:
        # current behaviour is ignoring proximity_days when target_date_str is yyyy format
        return asset_datetime_str[:4] == target_date_str

    asset_date = datetime.datetime.fromisoformat(asset_datetime_str).date()
    asset_doy = asset_date.timetuple().tm_yday
    if len_target_date_str == 10:
        target_doy = datetime.date.fromisoformat(target_date_str).replace(year=asset_date.year).timetuple().tm_yday
    elif len_target_date_str == 5:
        target_date_str = asset_datetime_str[:5] + target_date_str
        target_doy = datetime.date.fromisoformat(target_date_str).timetuple().tm_yday
    else:
        return False

    diff = abs(asset_doy - target_doy)
    wrapped_diff = min(diff, 365 - diff)
    return wrapped_diff <= proximity_days


def filter_people_data(people_data: list, birthday_bias: bool, favorite_people: bool) -> list:
    result = []
    for person in people_data:
        if len(person['name']) > 0 and ((not birthday_bias or person['birthDate']) and (not favorite_people
                                                                                        or person['isFavorite'])):
            result.append(person)
    return result


def find_featured_photo(assets: list, is_random: bool = False) -> str:
    if is_random:
        return random.choice(assets)['asset_id']

    highest_score = -1.0
    asset_id = ''

    for data in assets:
        if data['asset_score'] > highest_score:
            highest_score = data['asset_score']
            asset_id = data['asset_id']

    return asset_id


def fetch_server_version(root_url: str) -> dict:
    """
    Fetches assets from the Immich API
    Takes different API versions into account for compatibility
    """
    # This API call was only introduced with version 1.106.1, so it will fail
    # for older versions.
    # Initialize the version with the latest version without this API call
    version = {'major': 1, 'minor': 105, "patch": 1}

    try:
        # 1.118.x and up
        r = requests.get(root_url + 'server/version', **requests_kwargs)
        if r.status_code == 200:
            version = r.json()
            logging.info("Detected Immich server version (new API) %s.%s.%s", version['major'], version['minor'],
                         version['patch'])
            return version
    except requests.exceptions.RequestException as e:
        logging.warning("Failed to call new API endpoint: %s", e)

    try:
        # 1.106.x - 1.117.x
        r = requests.get(root_url + 'server-info/version', **requests_kwargs)
        if r.status_code == 200:
            version = r.json()
            logging.info("Detected Immich server version (old API) %s.%s.%s", version['major'], version['minor'],
                         version['patch'])
            return version
    except requests.exceptions.RequestException as e:
        logging.warning("Failed to call fallback API endpoint: %s", e)

    logging.info("Immich API not found or unsupported (version %s.%s.%s or older)", version['major'],
                 version['minor'], version['patch'])
    return version


def fetch_asset_image(root_url: str, asset_id: str) -> np.ndarray:
    r = requests.get(root_url + f'assets/{asset_id}/thumbnail', **requests_kwargs)
    if r.status_code == 200:
        return np.asarray(bytearray(r.content), dtype=np.uint8)
    return np.empty(0)


def fetch_assets_with_people(root_url, people_ids: list, fetch_chunk_size: int = 1000) -> list:
    """
    Fetches assets from the Immich API
    Uses the search/metadata call.
    """
    assets = []
    # prepare request body
    body = {}
    body['isOffline'] = 'false'
    body['type'] = 'IMAGE'
    body['personIds'] = people_ids
    body['withExif'] = 'true'

    # This API call allows a maximum page size of 1000
    number_of_assets_to_fetch_per_request_search = min(1000, fetch_chunk_size)
    body['size'] = number_of_assets_to_fetch_per_request_search

    # Initial API call, let's fetch our first chunk
    page = 1
    body['page'] = page
    r = requests.post(root_url + 'search/metadata', json=body, **requests_kwargs)
    r.raise_for_status()
    response_json = r.json()
    assets_received = response_json['assets']['items']
    logging.debug("Received %s assets with chunk %s", len(assets_received), page)

    assets += assets_received
    # If we got a full chunk size back, let's perfrom subsequent calls until we get less than a full chunk size
    while len(assets_received) == number_of_assets_to_fetch_per_request_search:
        page += 1
        body['page'] = page
        r = requests.post(root_url + 'search/metadata', json=body, **requests_kwargs)
        assert r.status_code == 200
        response_json = r.json()
        assets_received = response_json['assets']['items']
        logging.debug("Received %s assets with chunk %s", len(assets_received), page)
        assets += assets_received
    return assets


def fetch_people_data(root_url: str, person_ids=None, person_names=None, fetch_chunk_size: int = 1000) -> list:
    data = []
    if person_ids:
        for p_id in person_ids:
            p_data = fetch_person(root_url, person_id=p_id)
            if not p_data:
                continue

            data.append(p_data)

    if person_names:
        for p_name in person_names:
            p_data = fetch_person(root_url, person_name=p_name, fetch_chunk_size=fetch_chunk_size)
            if not p_data:
                continue

            data.append(p_data)

    if not (person_ids or person_names):
        data = fetch_people(root_url, fetch_chunk_size=fetch_chunk_size)

    return data


def fetch_people(root_url: str, fetch_chunk_size: int = 1000) -> list:
    """
    Fetches people from the Immich API
    Uses the /people call.
    """
    people = []
    # prepare request body
    body = {}

    # Initial API call, let's fetch our first chunk
    page = 1
    body['page'] = page

    # This API call allows a maximum page size of 1000
    number_of_assets_to_fetch_per_request_search = min(1000, fetch_chunk_size)
    body['size'] = number_of_assets_to_fetch_per_request_search

    body['withHidden'] = 'false'

    r = requests.get(root_url + 'people', params=body, **requests_kwargs)
    r.raise_for_status()
    response_json = r.json()

    people_received = response_json['people']
    total_people = response_json['total']
    total_hidden_people = response_json['hidden']
    has_next_page = response_json['hasNextPage']

    logging.debug("Received %s people with chunk %s", len(people_received), page)
    logging.info("Total amount of people: " + str(total_people))
    logging.info("Total amount of hidden people: " + str(total_hidden_people))

    people += people_received
    while has_next_page:
        page += 1
        body['page'] = page

        r = requests.get(root_url + 'people', params=body, **requests_kwargs)
        assert r.status_code == 200
        response_json = r.json()

        people_received = response_json['people']
        has_next_page = response_json['hasNextPage']
        logging.debug("Received %s people with chunk %s", len(people_received), page)

        people += people_received

    return people


def fetch_person(root_url: str, person_id: str = None, person_name: str = None, fetch_chunk_size: int = 1000) -> dict:
    global people_data_cache
    person_data = {}

    if person_id:
        try:
            r = requests.get(root_url + f'people/{person_id}', **requests_kwargs)
            r.raise_for_status()
            person_data = r.json()
        except requests.exceptions.HTTPError as e:
            if r.status_code == 400:
                logging.error("Bad request – invalid person_id: '%s' Did you mean to use -S or --people-names?",
                              person_id)
            else:
                raise

    if person_name:
        if not people_data_cache:
            people_data_cache = fetch_people(root_url, fetch_chunk_size=fetch_chunk_size)

        for p in people_data_cache:
            if p['name'] == person_name:
                person_data = p
                break

    return person_data


def update_person_featured(root_url, person_id: str, person_name: str, asset_id: str):
    global updated_people_names

    api_endpoint = 'people'
    data = {'featureFaceAssetId': asset_id}
    try:
        r = requests.put(root_url + api_endpoint + f'/{person_id}', json=data, **requests_kwargs)
        if r.status_code in [200, 201]:
            updated_people_names.append(person_name)
    except requests.exceptions.HTTPError as e:
        logging.error("Could not update the featured photo for id: '%s' name: '%s'", person_id, person_name)


if __name__ == "__main__":
    main()
    print(blur_score_cache)
