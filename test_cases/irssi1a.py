#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import subprocess
from collections import namedtuple

IrcMessage = namedtuple('IrcMessage', ['channel', 'timestamp', 'user', 'text'])


def channel_name(tail_header):
    '''Given a tail file header, extract the channel name.'''
    # tail prints headers of the current file, which look like this:
    # ==> /home/user/irclogs/server/#channel.log <==
    try:
        return tail_header.split('/')[-1].split('.')[0]
    except:
        print("error parsing line", tail_header)
        return None


def parse_msg(line, channel):
    line = line.strip()
    if line[5:].strip().startswith("-!-"):
        # this is a channel msg, e.g. user log off
        return None
    try:
        timestamp, rest = line.split('<', maxsplit=1)
        timestamp = timestamp.strip()
        user, rest = rest.split('>')
        return IrcMessage(channel, timestamp, user.strip(), rest.strip())
    except Exception as err:
        print("error parsing msg", line, err)
        return None


def run(user, channels=[], verbosity=0):
    channels = [c if c.startswith('#') else '#'+c for c in channels]
    print("will notify mentions of {}, or any activity in {}".format(
        user, ', '.join(channels)
    ))
    active_channel = ""
    while True:
        line = input()
        if not line:
            continue

        if line.startswith("==>"):
            active_channel = channel_name(line) or active_channel
            continue

        msg = parse_msg(line, active_channel)
        if not msg:
            if verbosity > 0:
                print('skipping line', line)
            continue

        notif_text = None
        # check if this is a message we're interested in
        if msg.user == user:
            continue
        if msg.channel in channels or msg.text.find(user) >= 0:
            notif_text = "{}@{}: {}".format(msg.user, msg.channel, msg.text)
        elif not msg.channel.startswith('#'):
            # a PM
            notif_text = "{}: {}".format(msg.user, msg.text)
        if notif_text:
            try:
                if verbosity > 0:
                    print("NOTIFYING", msg)
                subprocess.run(["growlnotify"], input=notif_text.encode('utf8'))
            except Exception as err:
                if verbosity > 0:
                    print("Exception: %s" % err)
        else:
            if verbosity > 0:
                print("SKIPPING", msg)


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Growl notifications for remote irssi sessions.')
    parser.add_argument('-u', '--user', type=str, help='[your] username.')
    parser.add_argument('-c', '--channels', type=str, nargs="+",
                        help='A list of channels. You will receive notifications of all\
                        messages in these channels.')
    parser.add_argument('-v', '--verbosity', action='store_true',
                        help='prints parsing info to stdout.')

    args = parser.parse_args()
    run(**vars(args))


if __name__ == "__main__":
    main()
