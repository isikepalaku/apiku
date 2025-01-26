from pathlib import Path

from phi.workspace.settings import WorkspaceSettings

#
# -*- Define workspace settings using a WorkspaceSettings object
# these values can also be set using environment variables or a .env file
#
ws_settings = WorkspaceSettings(
    # Workspace name: used for naming cloud resources
    ws_name="agent",
    # Path to the workspace root
    ws_root=Path(__file__).parent.parent.resolve(),
    # -*- Development env settings
    dev_env="dev",
    # -*- Development Apps
    dev_api_enabled=True,
    dev_db_enabled=True,
    # -*- Production env settings
    prd_env="prd",
    # -*- Production Apps
    prd_api_enabled=True,
    prd_db_enabled=True,
    # -*- AWS settings
    # Region for AWS resources
    aws_region="ap-southeast-2",
    # Availability Zones for AWS resources
    aws_az1="ap-southeast-2a",
    aws_az2="ap-southeast-2b",
    # Subnet IDs in the aws_region
    subnet_ids=["subnet-08ff0c275812e4764", "subnet-02b10a0259ed556a6"],
    # -*- Image Settings
    # Name of the image
    image_name="agent-api",
    # Repository for the image
    image_repo="phidata",
    # Build images locally
    build_images=True,
    push_images=True,
)
