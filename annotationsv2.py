import blenderproc as bproc
import os
import math
import time


def create_coco_annotations(camera, scene, output_dir,
                       object_id=False, new_position=False, new_rotation=False):

    print('camera file:', camera)

    bproc.init()

    # load the objects into the scene
    objs = bproc.loader.load_blend(scene)

    # Set some category ids for loaded objects
    for j, obj in enumerate(objs):
        obj.set_cp("category_id", j + 1)

    # define a light and set its location and energy level
    light = bproc.types.Light()
    light.set_type("POINT")
    light.set_location([5, -5, 5])
    light.set_energy(1000)

    # define the camera intrinsics
    bproc.camera.set_resolution(512, 512)

    # object manipulation
    if object_id != False:
        object = bproc.filter.one_by_attr(objs, "name", object_id)
        object.set_location(new_position)
        object.set_rotation_euler(new_rotation)

    # read the camera positions file and convert into homogeneous camera-world transformation
    with open(camera, "r") as f:
        for line in f.readlines():
            line = [float(x) for x in line.split()]
            position, euler_rotation = line[:3], line[3:6]
            matrix_world = bproc.math.build_transformation_mat(position, euler_rotation)
            bproc.camera.add_camera_pose(matrix_world)

    # activate normal rendering
    bproc.renderer.enable_normals_output()
    bproc.renderer.enable_segmentation_output(map_by=["category_id", "instance", "name"])

    # render the whole pipeline
    data = bproc.renderer.render()

    # Write data to coco file
    bproc.writer.write_coco_annotations(os.path.join(output_dir, 'coco_data'),
                                        instance_segmaps=data["instance_segmaps"],
                                        instance_attribute_maps=data["instance_attribute_maps"],
                                        colors=data["colors"],
                                        color_file_format="JPEG")

    # wait for image creation
    time.sleep(5)

    # visualize annotations
    os.system("python cocoviewer.py -i examples/advanced/coco_annotations/output/coco_data -a examples/advanced/coco_annotations/output/coco_data/coco_annotations.json")


def get_coco_annotations(scene_file, camera_dist, camera_height, camera_samples,
                     camera_filename, output_dir,
                     object_id=False, new_position=False, new_rotation=False):

    # define first camera position
    camera_positions = [[camera_dist, 0, camera_height, math.atan(camera_dist / camera_height), 0, math.pi / 2]]

    # define remaining camera positions
    for i in range(1, camera_samples):
        last_position = camera_positions[-1]
        next_position = [camera_dist * math.cos((2 * i * math.pi) / camera_samples),
                         camera_dist * math.sin((2 * i * math.pi) / camera_samples), last_position[2], last_position[3],
                         last_position[4], last_position[5] + 2 * math.pi / camera_samples]
        camera_positions.append(next_position)

    # write camera positions in camera file
    with open(camera_filename, 'w') as camera_file:
        for position in camera_positions:
            line = ' '.join(str(e) for e in position)
            camera_file.write(line)
            camera_file.write('\n')

    camera_file.close()

    # call a function to generate the coco annotations
    create_coco_annotations(camera_filename, scene_file, output_dir,
                       object_id, new_position, new_rotation)


get_coco_annotations("/home/fsmatilde/fsmatilde_ext/OceanTest1.blend", 1000, 200, 4,
                 camera_filename="examples/resources/camera_positions",
                 output_dir="examples/advanced/coco_annotations/output")
                 # object_id = "Suzanne", new_position=[1,2,3], new_rotation=[1,1,0])