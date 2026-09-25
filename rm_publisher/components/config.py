import argparse

def parse_args(args):
    conf = {}
    if args.rm:
        conf['robomaster'] = args.rm
    if args.offset:
        conf['offset'] = list(map(int, args.offset.split(',')))
    if args.map:
        conf['map_size'] = args.map
    if args.interpolated:
        conf['interpolated'] = args.interpolated
    return conf

def get_args():
    parser = argparse.ArgumentParser(description='Robomaster')
    parser.add_argument("--rm", type=str, required=False, help="name of the robomaster")
    parser.add_argument("--offset", type=str, required=False, help='a list with three int specifying the x,y and z offset')
    parser.add_argument("--map", type=int, required=False, help='an int for the map size')
    parser.add_argument("--interpolated", type=bool, required=False, help='bool to specify velocity type to publish')
    args = parser.parse_args()
    return args