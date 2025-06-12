import os

os.system('ls -ltr /dev/disk/by-id/ > '+os.environ['HOME']+'/tmp.txt')
with open(os.path.join(os.environ['HOME'], 'tmp.txt')) as f:
  lines = f.readlines()
ids = set()
for line in lines:
  if '->' in line:
    path = line.split('->')[1].strip().split('/')[-1]
    if len(path) == 3 and path != 'sda':
      ids.add(path)
assert len(ids) == 1, ids
ids = list(ids)
os.system('sudo mkdir -p /mnt/disks/data')
os.system('sudo umount /mnt/disks/data')
os.system(f'sudo mount -o discard,defaults /dev/{ids[0]} /mnt/disks/data')
os.system(f'unlink /home/{os.environ["REMOTE_USER"]}/buckets/data')
os.system('ln -s /mnt/disks/data/ /home/{os.environ["REMOTE_USER"]}/buckets/data')
