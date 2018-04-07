import boto.ec2, sys
# AWS_ACCESS_KEY_ID
AKID='AKIAIH7J5O3U4JJVKY7Q'
# AWS_SECRET_ACCESS_KEY
ASAK='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'
# Region string
REGION='eu-west-2'
 
def print_usage(args):
  print ("Usage:", args[0], "stop|start <instance name>")
  sys.exit(1)
 
def usage(args):
  arg1 = ['stop', 'start']
  if not len(args) == 3:
    print_usage(args)
  else:
    if not args[1] in arg1:
      print_usage(args)
    else:
      return args[2]
 
myinstance = usage(sys.argv)
conn = boto.ec2.connect_to_region(REGION, aws_access_key_id=AKID, aws_secret_access_key=ASAK)
if sys.argv[1] == 'start':
  print ("Attempt on starting the instance : %s" % str(myinstance))
  try:
    inst = conn.get_all_instances(filters={'tag:Name': myinstance})[0].instances[0]
  except IndexError:
    print ('Error:', myinstance, 'not found!')
    sys.exit(1)
  if not inst.state == 'running':
    print ("Instance %s started[OK]" %  str(myinstance))
    inst.start()
  else:
    print ('Error:', myinstance, 'is already running or starting up!')
    sys.exit(1)
 
if sys.argv[1] == 'stop':
  print ("Attempt on stopping the instance : %s" % str(myinstance))
  try:
    inst = conn.get_all_instances(filters={'tag:Name': myinstance})[0].instances[0]
  except IndexError:
    print ('Error:', myinstance, 'not found!')
    sys.exit(1)
  if inst.state == 'running':
    print ("Instance %s stopped[OK]" %  str(myinstance))
    inst.stop()
  else:
    print ('Error:', myinstance, 'already stopped or stopping')
    sys.exit(1)