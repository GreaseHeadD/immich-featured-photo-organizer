import pytest
import datetime
from math import exp
import immich_featured_photo_organizer

PEOPLE_DATA = [
    {'id': '393da4b3-d26e-4185-8a11-60db04b8b900', 'name': 'John Doe', 'birthDate': '2003-01-02',
     'isHidden': False, 'isFavorite': True},
    {'id': '9740d76f-5a9a-4928-bffd-0c46424a5abb', 'name': 'Jane Doe', 'birthDate': '2007-04-06',
     'isHidden': False, 'isFavorite': False},
    {'id': 'f914f57e-8400-41b1-b52b-49b6291ea66b', 'name': 'Jake', 'birthDate': None,
     'isHidden': False, 'isFavorite': True},
    {'id': '45eedb1f-b175-475c-9293-392161648ce5', 'name': 'Olivia', 'birthDate': None,
     'isHidden': False, 'isFavorite': False},
    {'id': '0b6c42f4-29e4-42c4-83e7-a0bb8a04e4a7', 'name': '', 'birthDate': None,
     'isHidden': False, 'isFavorite': False},
    {'id': '0b6c42f4-29e4-42c4-83e7-a0bb8a04e4a7', 'name': '', 'birthDate': None,
     'isHidden': False, 'isFavorite': True}
]


@pytest.fixture
def person_data():
    return {'person_id': '393da4b3-d26e-4185-8a11-60db04b8b900', 'person_name': 'John Doe',
            'person_birthDate': '2003-01-02', 'assets': []}


@pytest.fixture
def asset_data():
    return {"id": "92ae2001-a917-4fc9-a2dd-8671a99d39a5", "deviceAssetId": "IMG_2307.JPG",
            "ownerId": "0b431918-436e-45d7-889f-2ff0c7138080",
            "owner": {"id": "0b431918-436e-45d7-889f-2ff0c7138080", "email": "omitted", "name": "omitted",
                      "profileImagePath": "", "avatarColor": "amber",
                      "profileChangedAt": "2024-09-28T22:29:44.193978+00:00"}, "deviceId": "Library Import",
            "libraryId": "116fcd5c-11bc-42e1-89a2-c6ecbb6be7e3", "type": "IMAGE",
            "originalPath": "/import/camera/2025-06/IMG_2307.JPG", "originalFileName": "IMG_2307.JPG",
            "originalMimeType": "image/jpeg", "thumbhash": "xQcGDYAWl3l4l2iAi5d4eYCpGWQJ",
            "fileCreatedAt": "2025-06-08T00:15:04.940Z", "fileModifiedAt": "2025-06-07T23:15:05.402Z",
            "localDateTime": "2025-06-08T01:15:04.940Z", "updatedAt": "2025-07-13T22:01:42.556Z", "isFavorite": False,
            "isArchived": False, "isTrashed": False, "visibility": "timeline", "duration": "0:00:00.00000",
            "exifInfo": {"make": "Canon", "model": "Canon EOS 250D", "exifImageWidth": 6000, "exifImageHeight": 4000,
                         "fileSizeInByte": 5102306, "orientation": "1",
                         "dateTimeOriginal": "2025-06-08T00:15:04.94+00:00",
                         "modifyDate": "2025-06-07T23:15:05.402+00:00", "timeZone": "UTC+1", "lensModel": None,
                         "fNumber": 0, "focalLength": 0, "iso": 800, "exposureTime": "1/160", "latitude": None,
                         "longitude": None, "city": None, "state": None, "country": None, "description": "",
                         "projectionType": None, "rating": 0}, "livePhotoVideoId": None, "tags": [], "people": [
            {"id": "393da4b3-d26e-4185-8a11-60db04b8b900", "name": "John Doe", "birthDate": "2000-03-08",
             "thumbnailPath":
                 "/photos/thumbs/0b431918-436e-45d7-889f-2ff0c7138080/39/3d/393da4b3-d26e-4185-8a11-60db04b8b900.jpeg",
             "isHidden": False, "isFavorite": False, "updatedAt": "2025-07-24T00:28:59.345958+00:00", "faces": [
                {"id": "5fe27ece-ec06-4f44-a0ca-e63b0898e9f2", "imageHeight": 1440, "imageWidth": 2160,
                 "boundingBoxX1": 1262, "boundingBoxX2": 1454, "boundingBoxY1": 696, "boundingBoxY2": 957,
                 "sourceType": "machine-learning"}]}], "unassignedFaces": [],
            "checksum": "cEfoq16IBYPN8w8NQHxk8Dtw6QM=", "stack": None, "isOffline": False, "hasMetadata": True,
            "duplicateId": None, "resized": True}


def test_calculate_face_size(asset_data, person_data, monkeypatch):
    def mock_calculate_real_face_size(x1, x2, y1, y2, resize_factor):
        assert x1 == 1262
        assert x2 == 1454
        assert y1 == 696
        assert y2 == 957
        assert resize_factor == pytest.approx((6000 * 4000) / (2160 * 1440))
        return 42.0

    monkeypatch.setattr("immich_featured_photo_organizer.calculate_real_face_size", mock_calculate_real_face_size)

    result = immich_featured_photo_organizer.calculate_face_size(asset_data, person_data['person_id'])
    assert result == 42.0


@pytest.mark.parametrize(
    "now_str, date_str, reversed_recency, decay_rate, expected_result",
    [
        # 1 day ago, normal direction
        ("2024-01-01", "2023-12-31", False, 0.004, exp(-0.004 * 1)),
        # 2 days ago, reversed direction
        ("2024-01-01", "2023-12-30", True, 0.004, exp(0.004 * 2)),
        # same day, result should be exp(0) == 1.0
        ("2024-01-01", "2024-01-01", False, 0.004, 1.0),
        # future date: should be clamped to 0 days ago
        ("2024-01-01", "2024-01-02", False, 0.004, 1.0),
        # different decay rate
        ("2024-01-01", "2023-12-31", False, 0.01, exp(-0.01 * 1)),
    ]
)
def test_calculate_datetime_recency(now_str, date_str, reversed_recency, decay_rate, expected_result):
    now = datetime.datetime.fromisoformat(now_str)
    result = immich_featured_photo_organizer.calculate_datetime_recency(
        date_str,
        curr_datetime=now,
        reversed_recency=reversed_recency,
        decay_rate=decay_rate
    )
    assert result == pytest.approx(expected_result, rel=1e-6)


@pytest.mark.parametrize(
    "datetime_str, target_date_str, sigma, expected_result",
    [
        # Exact match => max proximity
        ("2024-01-01", "2024-01-01", 7.0, 1.0),
        # 1 day away with default sigma
        ("2024-01-02", "2024-01-01", 7.0, exp(-(1 ** 2) / (2 * 7.0 ** 2))),
        # 3 days away with sigma 2
        ("2024-01-04", "2024-01-01", 2.0, exp(-(3 ** 2) / (2 * 2.0 ** 2))),
        # 0.5 day difference
        ("2024-01-01T12:00:00", "2024-01-01T00:00:00", 7.0, exp(-(0.5 ** 2) / (2 * 7.0 ** 2))),
        # 10 days away with sigma 5
        ("2024-01-11", "2024-01-01", 5.0, exp(-(10 ** 2) / (2 * 5.0 ** 2))),
        # Test default sigma fallback (None)
        ("2024-01-02", "2024-01-01", None, exp(-(1 ** 2) / (2 * 1.0 ** 2))),
    ]
)
def test_calculate_datetime_proximity(datetime_str, target_date_str, sigma, expected_result):
    result = immich_featured_photo_organizer.calculate_datetime_proximity(datetime_str, target_date_str, sigma)
    assert result == pytest.approx(expected_result, rel=1e-6)


@pytest.mark.parametrize(
    "datetime_str, target_mm_dd, sigma, expected_result",
    [
        # Exact match (Jan 1)
        ("2024-01-01", "01-01", 7.0, 1.0),

        # 1 day apart (Jan 2 vs Jan 1)
        ("2024-01-02", "01-01", 7.0, exp(-1 ** 2 / (2 * 7.0 ** 2))),

        # 3 days apart with sigma = 2
        ("2024-01-04", "01-01", 2.0, exp(-3 ** 2 / (2 * 2.0 ** 2))),

        # Leap year edge case (Feb 29)
        ("2024-02-29", "03-01", 7.0, exp(-1 ** 2 / (2 * 7.0 ** 2))),

        # Cross-year circular test (Dec 31 vs Jan 1)
        ("2024-12-31", "01-01", 7.0, exp(-1 ** 2 / (2 * 7.0 ** 2))),

        # Long distance across wraparound (Dec 25 vs Jan 1 = 7 days)
        ("2024-12-25", "01-01", 7.0, exp(-7 ** 2 / (2 * 7.0 ** 2))),
    ],
    ids=[
        "exact match",
        "1 day apart",
        "3 days, sigma=2",
        "leap year Feb 29 vs Mar 1",
        "wraparound Dec 31 vs Jan 1",
        "wraparound Dec 25 vs Jan 1"
    ]
)
def test_calculate_datetime_proximity_yearless(datetime_str, target_mm_dd, sigma, expected_result):
    result = immich_featured_photo_organizer.calculate_datetime_proximity_yearless(datetime_str, target_mm_dd, sigma)
    assert result == pytest.approx(expected_result, rel=1e-6)


@pytest.mark.parametrize(
    "asset_dt, target_date, expected_min",
    [
        ("2024-01-01", "", 1.0),  # empty target_date
        ("", "2024-01-01", 1.0),  # empty asset_dt
        ("2024-01-01", "2024-01-01", 0.99),  # length 10 exact match -> near 1.0
        ("2024-01-02", "01-01", 0.6),  # length 5 -> proximity yearless
        ("2024-01-01", "2024", 0.5),  # length 4 -> special case with appended '-07-02'
        ("2024-01-01", "xxx", 1.0),  # unexpected length -> 1.0
    ]
)
def test_calculate_date_bias(asset_dt, target_date, expected_min):
    result = immich_featured_photo_organizer.calculate_date_bias(asset_dt, target_date)
    assert 0 <= result <= 1
    assert result >= expected_min


@pytest.mark.parametrize("recency, reversed_recency, datebias, birthdate_str, oldest_str, expected", [
    # Case 1: recency_bias active
    (True, False, '', "1990-05-20", '', "2025-05-20"),  # birthday in the past this year
    (True, False, '', "1990-12-31", '', "2024-12-31"),  # birthday in the future -> shift to last year

    # Case 2: recency_bias_reversed active
    (False, True, '', "1990-05-20", "2000-06-01", "2001-05-20"),  # birthday before oldest -> shift to next year
    (False, True, '', "1990-12-31", "2000-12-31", "2000-12-31"),  # birthday same day

    # Case 3: year-only date_bias
    (False, False, '2010', "1990-05-20", '', "2010-05-20"),

    # Case 4: fallback yearless
    (False, False, '', "1990-05-20", '', "05-20"),
])
def test_get_adjusted_birthday(monkeypatch, recency, reversed_recency, datebias, birthdate_str, oldest_str, expected):
    monkeypatch.setattr("immich_featured_photo_organizer.current_datetime",
                        datetime.datetime(2025, 7, 2, hour=12, minute=15, second=58,
                                          tzinfo=datetime.timezone.utc))
    result = immich_featured_photo_organizer.get_adjusted_birthday(birthdate_str, oldest_datetime_str=oldest_str,
                                                                   is_recency=recency,
                                                                   is_recency_reversed=reversed_recency,
                                                                   biased_date=datebias)
    assert result == expected


@pytest.mark.parametrize("adjusted_birthday, expected_bias, expected_func", [
    ("2025-06-20", 0.8, "proximity"),  # len == 10 -> full proximity
    ("05-20", 0.9, "yearless"),  # len == 5 -> yearless
    ("1990", 0.8, "proximity"),  # len == 4 -> proximity
    ("", 1.0, None),  # fallback
])
def test_calculate_birthday_bias(monkeypatch, adjusted_birthday, expected_bias, expected_func):
    monkeypatch.setattr("immich_featured_photo_organizer.get_adjusted_birthday",
                        lambda *_args, **_kwargs: adjusted_birthday)

    called = {"proximity": False, "yearless": False}

    def fake_proximity(asset_dt, target_dt, sigma):
        called["proximity"] = True
        return 0.8

    def fake_yearless(asset_dt, target_mmdd, sigma):
        called["yearless"] = True
        return 0.9

    monkeypatch.setattr("immich_featured_photo_organizer.calculate_datetime_proximity", fake_proximity)
    monkeypatch.setattr("immich_featured_photo_organizer.calculate_datetime_proximity_yearless", fake_yearless)

    result = immich_featured_photo_organizer.calculate_birthday_bias("2025-07-02", "1990-05-20",
                                                                     oldest_datetime_str="2000-01-01", sigma=7.0)
    assert result == expected_bias

    if expected_func:
        assert called[expected_func]
        assert not any(v for k, v in called.items() if k != expected_func)
    else:
        assert not any(called.values())


@pytest.mark.parametrize("asset_date, target_date, proximity_days, expected", [
    # birthday tests
    ("2025-05-20T00:00:00.000+00:00", "2000-05-20", 0, True),  # Exact match
    ("2025-05-21T00:00:00.000+00:00", "2000-05-20", 0, False),  # Exact match 1 day off
    ("2025-05-21T12:34:56.789+00:00", "2000-05-20", 1, True),  # 1 day after
    ("2025-05-23T23:59:59.999+00:00", "2000-05-20", 2, False),  # 3 days apart
    ("2025-01-01T10:00:00.000+00:00", "2000-12-31", 1, True),  # Wraparound: Jan 1 vs Dec 31
    ("2025-12-31T23:59:59.000+00:00", "2000-01-01", 1, True),  # Wraparound: Dec 31 vs Jan 1
    ("2025-07-01T00:00:00.000+00:00", "2000-07-03", None, False),  # 2 days apart, None -> proximity_days = 1
    ("2025-07-02T00:00:00.000+00:00", "2000-07-03", None, True),  # 1 day apart
    # regular dates
    ("2025-02-21T00:00:00.000+00:00", "2000", None, False),
    ("2025-06-10T00:00:00.000+00:00", "2025", None, True),
    ("2024-12-25T00:00:00.000+00:00", "12-25", None, True),
    ("2025-01-22T00:00:00.000+00:00", "01-23", 0, False),
])
def test_is_date_asset(asset_date, target_date, proximity_days, expected):
    result = immich_featured_photo_organizer.is_date_asset(asset_date, target_date, proximity_days)
    assert result == expected


@pytest.mark.parametrize("birthday_bias, favorite_people, with_names, without_names, expected", [
    (False, False, True, False, PEOPLE_DATA[:-2]),
    (True, False, True, False, PEOPLE_DATA[:2]),
    (False, True, True, False, PEOPLE_DATA[0:3:2]),
    (True, True, True, False, PEOPLE_DATA[:1]),
    (False, False, False, True, PEOPLE_DATA[-2:]),
    (False, False, False, False, PEOPLE_DATA),
    (False, True, False, True, PEOPLE_DATA[-1:]),
])
def test_filter_people_data(birthday_bias, favorite_people, with_names, without_names, expected):
    result = immich_featured_photo_organizer.filter_people_data(PEOPLE_DATA, birthday_bias, favorite_people, with_names, without_names)
    print(result)
    assert result == expected
