provider "aws"{
  region="ap-south-1"
  profile="default"
}
resource "aws_instance" "os1"{
   ami="ami-0ad704c126371a549"
   instance_type="t2.micro"
}
resource "aws_ebs_volume" "storage"{
   availability_zone=aws_instance.os1.availability_zone
   size=5
}
resource "aws_volume_attachment" "storage_attach"{
       device_name="/dev/sdh"
       volume_id=aws_ebs_volume.storage.id
       instance_id=aws_instance.os1.id
}
