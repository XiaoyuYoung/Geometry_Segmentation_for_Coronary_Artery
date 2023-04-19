import SimpleITK as sitk
import numpy as np

def get_yaml(file):
    import yaml
    with open(file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def get_medical_image(path):
    reader = sitk.ReadImage(path)
    array = sitk.GetArrayFromImage(reader)
    spacing = reader.GetSpacing()
    origin = reader.GetOrigin()
    direction = reader.GetDirection()
    image_type = reader.GetPixelID()
    return array, {'origin': origin, 'spacing': spacing, 'direction': direction, 'type': image_type}


def save_medical_image(array, target_path, param=None):
    if isinstance(array, sitk.Image):
        image = array
        sitk.WriteImage(image, target_path, True)
        return

    image = sitk.GetImageFromArray(array, isVector=False)

    if 'direction' in param: image.SetDirection(param['direction'])
    if 'spacing' in param: image.SetSpacing(param['spacing'])
    if 'origin' in param: image.SetOrigin(param['origin'])

    if 'type' not in param:
        sitk.WriteImage(image, target_path, True)
    else:
        sitk.WriteImage(sitk.Cast(image, param['type']), target_path, True)


def get_json(file):
    import json
    with open(file, 'r', encoding='utf-8') as f:
        dicts = json.load(f)
    return dicts


def get_csv(file, delimiter=','):
    import csv
    with open(file, 'r') as f:
        reader = csv.reader(f, delimiter=delimiter)
        result = list(reader)
    return result


def save_csv(file, rows, headers=None, delimiter=','):
    import csv
    with open(file, 'w', newline='') as f:
        writer = csv.writer(f, delimiter=delimiter)
        if headers is not None:
            writer.writerow(headers)
        writer.writerows(rows)


def norm_zero_one(array, span=None):
    array = np.asarray(array).astype(np.float32)
    if span is None:
        mini = array.min()
        maxi = array.max()
    else:
        mini = span[0]
        maxi = span[1]
        array[array < mini] = mini
        array[array > maxi] = maxi

    range = maxi - mini

    def norm(x):
        return (x - mini) / range

    return np.asarray(list(map(norm, array))).astype(np.float32)

def save_mps(points, target_path, offset=(0, 0, 0), direction=(1, 0, 0, 0, 1, 0, 0, 0, 1)):
    points = np.asarray(points)
    direction = """
     <IndexToWorld type="Matrix3x3" m_0_0="{}" m_0_1="{}" m_0_2="{}" m_1_0="{}" m_1_1="{}" m_1_2="{}" m_2_0="{}" m_2_1="{}" m_2_2="{}"/>
     """.format(direction[0], direction[1], direction[2],
                direction[3], direction[4], direction[5],
                direction[6], direction[7], direction[8])

    offset = """<Offset type="Vector3D" x="{}" y="{}" z="{}"/>
    """.format(offset[0], offset[1], offset[2])

    minx, miny, minz = points[:, 0].min(), points[:, 1].min(), points[:, 2].min()
    maxx, maxy, maxz = points[:, 0].max(), points[:, 0].max(), points[:, 0].max()
    bounds = """
    <Bounds>
        <Min type="Vector3D" x="{}" y="{}" z="{}"/>
        <Max type="Vector3D" x="{}" y="{}" z="{}"/>
    </Bounds>
    """.format(minx, miny, minz, maxx, maxy, maxz)

    strpoints = ""
    for index, point in enumerate(points):
        x, y, z = point
        strpoints += """
        <point>
            <id>{}</id>
            <specification>0</specification>
            <x>{}</x>
            <y>{}</y>
            <z>{}</z>
        </point>
        """.format(index, x, y, z)

    mps = """
    <?xml version="1.0" encoding="UTF-8"?>
    <point_set_file>
    <file_version>0.1</file_version>
    <point_set>
        <time_series>
            <time_series_id>0</time_series_id>
            <Geometry3D ImageGeometry="false" FrameOfReferenceID="0">
                {}
                {}
                {}
            </Geometry3D>
            {}
        </time_series>
    </point_set>
    </point_set_file>
    """.format(direction, offset, bounds, strpoints)

    with open(target_path, 'w', encoding='utf-8')as file:
        file.write(mps)
