# Immich Screenshots Organizer

This is a python script designed to automatically set the featured photo of a person in [Immich](https://immich.app/) to the photo with the highest quality.
Because the featured photo is automatically set for a person when they are added, setting this property can be a time consuming task for large libraries.

__Current compatibility:__ Immich v1.136.x and up

# Table of Contents
1. [Usage (Bare Python Script)](#bare-python-script)
2. [How It Works](#how-it-works)
3. [Performance](#performance)
4. [Notes](#notes)

## Usage
### Bare Python Script
1. Download the script and its requirements
    ```bash
    curl https://raw.githubusercontent.com/GreaseHeadD/immich-featured-photo-organizer/main/immich_featured_photo_organizer.py -o immich_featured_photo_organizer.py
    curl https://raw.githubusercontent.com/GreaseHeadD/immich-featured-photo-organizer/main/requirements.txt -o requirements.txt
    ```
2. Install requirements
    ```bash
    pip3 install -r requirements.txt
    ```
3. Run the script
    ```
    usage: immich-featured-photo-organizer.py api_url api_key [options]
    ```

__Plain example without optional arguments:__
```bash
python3 ./immich-featured-photo-organizer.py https://immich.mydomain.com thisIsMyApiKeyCopiedFromImmichWebGui
```

__API key__

As of Immich release 1.1360.0 the minimum required permissions are: `all`. This is due to search calls not having it's own permission scope, however this scope will be added in the future.

__Options__

| Option                                    | Description                                                       |
| ----------------------------------------- | ----------------------------------------------------------------- |
| `-h`, `--help`                            | Show the help message and exit.                                   |
| `-u`, `--unattended`                      | Do not ask for user confirmation after identifying people. Set this flag to run script as a cronjob. (default: `False`) |
| `-s`, `--people-ids` `PEOPLE_IDS`           | Select featured photos for specific people using ids. Format: `-s id1 id2 id3`, `-s id1 -s id2 -s id3` |
| `-S`, `--people-names` `PEOPLE_NAMES`       | Select featured photos for specific people using names. Format: `-S name1 "name2 surname" name3`, `-S name1 -S "name2 surname" -S name3` |
| `-f`, `--favorite-people`                     | Select featured photos for favorite people. |
| `-r`, `--random-mode`                     | Select featured photos randomly. |
| `-p`, `--low-pixel-mode`                  | Favor low resolution photos. |
| `-b`, `--detect-blur`                     | Detect if images are blurry, favoring sharpness. |
| `-B`, `--detect-blur-reversed`            | Detect if images are blurry, favoring blurriness. |
| `-t`, `--recency-bias`                    | Take image recency into account, favoring newest. |
| `-T`, `--recency-bias-reversed`           | Take image recency into account, favoring oldest. |
| `-d`, `--date-bias` `yyyy-mm-dd` or `mm-dd` or `yyyy` | Favor pictures taken around a certain date, yearless or a specific year. |
| `-n`, `--birthday-bias`                   | Favor photod around a person's birthday. Can be combined with `-r`, `-t`, `-T`. |
| `-g`, `--sigma-days SIGMA_DAYS`                   | Set the sigma days when using `-d`, `-D` or `-n`. |
| `-C`, `--fetch-chunk-size FETCH_CHUNK_SIZE` | Maximum number of assets and people to fetch per API call. (default: `1000`) |
| `-l`, `--log-level {CRITICAL,ERROR,WARNING,INFO,DEBUG}` | Set the logging level. (default: `INFO`) |

__note: the `--people-ids`, `--people-names` and `--favorite-people` options CAN be mixed together__

### Examples
Select the most recent non-blurry photo for everyone (resource intensive):
```bash
immich-featured-photo-organizer.py api_url api_key -bt
```

Select a random birthday photo for everyone:
```bash
immich-featured-photo-organizer.py api_url api_key -rn
```

Select a photo for "My Mom" and "My Dad" around christmas:
```bash
immich-featured-photo-organizer.py api_url api_key -S "My Mom" "My Dad" -d 12-25
```

Select a low-pixel photo for Mark:
```bash
immich-featured-photo-organizer.py api_url api_key -p -S Mark
```

## How it works

The script utilizies [Immich's REST API](https://immich.app/docs/api/) to query all people indexed by Immich, then 
fetches the assets in which a certain person is present. A score is calculated for each asset belonging to a person and the asset with the highest score is set as featured photo.

The score is calculated as follows: `asset_score = face_size * blur_score * date_score`. If the `--random-mode` option is set, score calculations will be skipped as will the `blur-score` in absence of `--detect-blur` or `--detect-blur-reversed`, and the `date_score` in absence of any of the date-related options.

### face_size
The `face_size` is based on the area of the facial recognition bounding box of the first face found (in case of collages, screenshots) belonging to the person. This number is then scaled to reflect the dimensions of the original photo. default is 1.0

### blur_score
When blur detection is enabled the laplacian variance is calculated to determine a blur score. A high `blur_score` means that the photo is not blurry. default is 1.0

### date_score
The `date_score` is calculated based on the date options that are set. 

The recency bias score is calculated using an exponential decay function with a decay speed of ~180 days, meaning the score is half the highest score when the photo datetime is 180 days from the current datetime.

The date bias score is calculated using a probability density function with a sigma of 7 days, meaning photos with a date that fall in the range of 7 days before the target date and 7 days after the target date get favored the most. The sigma can be set with `--sigma-days`.

The birthday bias is calculated as a yearless date bias. If a person does not have a birthday set the person will be skipped. If you want the birthday bias to only favor the most recent or oldest photos, combine it with a `-t` or `-T` respectively. If you want it to favor a specific year, use `-d yyyy`.

## Performance

The script was tested on a library with 560 people. If specifying specific people `--people_ids` might be faster than `--people_names` due to making less API calls. However specifying people is much faster than not doing so. The script should be the fastest under `--random-mode` since no scores need to be calculated.

### Standard performance

### Blur detection performance
The performance when blur detection is enabled is noticably slower than standard performance. This is due to needing to fetch the actual asset image, resizing it and then calculating the score.

## Notes

- This scripts only processes people that have a name, if you wish to process all detected people please open an issue.

- Only assets with timeline visibility are processed. Archived, locked, hidden and trashed assets are not included.

- Because of the way people data is structured in Immich, there is no way of knowing the asset id of the current featured photo of a person. Thus there is no way of determining whether the new featured photo is the same as the old featured photo. A face score for the old featured photo can consequently also not be calculated.

- As of release 1.360.0, API calls to routes with a non-specific scope require the API key to have the `all` permission. The missing scopes may be added at a later moment to Immich.
