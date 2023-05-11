import kfp
import kfp.dsl as dsl
from kfp import components

"""
This code demonstrates an example of two Kubeflow components sharing a path in a volume. In this example, the first component, 
`write_to_volume`, writes a simple message into a file named `file.txt` located in the `/mnt` path of a persistent volume created by `create-pvc`. 
The second component, `read_from_volume`, reads the content of the `file.txt` file from the same path and prints it.

The `dsl.VolumeOp` class creates a persistent volume claim (PVC) named `my-pvc` using `modes=dsl.VOLUME_MODE_RWM` which allows for both read and write operations to the volume.
The `write_to_volume` component uses `add_pvolumes` to specify the `/mnt` path on the volume created by `create-pvc`.
Then `read_from_volume` component uses `add_pvolumes` to mount the same volume at `/mnt` and reads the content of the `file.txt` file written by the previous component.

Finally, the pipeline is compiled using `kfp.compiler.Compiler().compile()` and submitted for execution to Kubeflow using the `Client().run_pipeline()` method. 
This code can serve as a starting point for users who want to create Kubeflow components that share a path in a volume.
"""


# Create component
@components.create_component_from_func
def write_to_volume():
    with open("/mnt/file.txt", "w") as file:
        file.write("Hello world 5")


# Create component
@components.create_component_from_func
def read_from_volume():
    with open("/mnt/file.txt", "r") as file:
        contents = file.read()
        print(f"File contents would be  : {contents}")


# Define pipeline
@dsl.pipeline(name="volumeop-basic", description="A Basic Example on VolumeOp Usage.")
def volumeop_basic(size: str = "1Gi"):
    vop = dsl.VolumeOp(
        name="create-pvc", resource_name="my-pvc", modes=dsl.VOLUME_MODE_RWM, size=size
    )

    write_op = write_to_volume().add_pvolumes({"/mnt": vop.volume})
    read_op = read_from_volume().add_pvolumes({"/mnt": write_op.pvolume})


if __name__ == "__main__":
    # Compile the pipeline
    pipeline_func = volumeop_basic
    pipeline_filename = pipeline_func.__name__ + ".yaml"
    kfp.compiler.Compiler().compile(pipeline_func, pipeline_filename)

    # Connect to the Kubeflow Pipeline and submit the pipeline for execution
    client = kfp.Client()
    experiment_name = "VolumeOp Basic Pipeline"
    experiment = client.create_experiment(experiment_name)
    run_name = "run"
    run = client.run_pipeline(experiment.id, run_name, pipeline_filename, {})
