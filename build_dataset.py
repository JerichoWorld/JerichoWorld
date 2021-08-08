from jericho import *
from jericho.template_action_generator import TemplateActionGenerator
import json
from jericho.util import *#util as jrutl
import random

import defines_updated
from copy import deepcopy


def load_attributes():
    global attributes
    global readable
    global MOVE_ACTIONS
    MOVE_ACTIONS = 'north/south/west/east/northwest/southwest/northeast/southeast/up/down/enter/exit'.split('/')

    with open('symtables/readable_tables.txt', 'r') as f:
        readable = [str(a).strip() for a in f]

    attributes = {}
    attr_vocab = set()
    for gn in readable:
        attributes[gn] = {}

        with open('symtables/' + gn + '.out', 'r') as f:
            try:
                for line in f:
                    if "attribute" in line.lower():
                        split = line.split('\t')
                        if len(split) < 2:
                            split = line.split()
                            idx, attr = int(split[1]), split[2]
                            if len(split) < 2:
                                continue
                        else:
                            idx, attr = int(split[0].split(' ')[1]), split[1]
                        if '/' in attr:
                            attr = attr.split('/')[0]
                        attributes[gn][idx] = attr.strip()
                        attr_vocab.add(attr.strip())

            except UnicodeDecodeError:
                print("Decode error:", gn)
                continue


def tree_to_triple(cur_loc, you, sub_tree, prev_act, prev_loc, game_name):
    triples = set()

    triples.add(('you', 'in', cur_loc.name))
    if prev_act.lower() in MOVE_ACTIONS:
        triples.add((cur_loc.name, prev_act.replace(' ', '_'), prev_loc.name))

    if prev_act.lower() in defines.ABBRV_DICT.keys():
        prev_act = defines.ABBRV_DICT[prev_act.lower()]
        triples.add((cur_loc.name, prev_act.replace(' ', '_'), prev_loc.name))

    for obj in sub_tree:
        if obj.num == you.num:
            continue
        elif obj.parent == you.num:
            triples.add(('you', 'have', obj.name))
        elif obj.parent == cur_loc.num:
            triples.add((obj.name, 'in', cur_loc.name))
        else:
            cur_parent = [a.name for a in sub_tree if a.num == obj.parent]

            triples.add((obj.name, 'in', cur_parent[0]))

        if game_name in readable:
            cur_attrs = attributes[game_name]
            obj_attrs = obj.attr
            for oatr in obj_attrs:
                if oatr in cur_attrs.keys():
                    triples.add((obj.name, 'is', cur_attrs[oatr].lower()))

    return list(triples)


def graph_diff(graph1, graph2):
    graph1 = set(graph1)
    graph2 = set(graph2)
    return list((graph2 - graph1))

def identify_interactive_objects(env, obs_desc, inv_desc, state, gamename):
    surr_objs, inv_objs = set(), set()
    state = env.get_state()

    # Extract objects from observation
    obs = extract_objs(obs_desc)
    surr_objs = surr_objs.union(obs)

    # Extract objects from inventory description
    inv = extract_objs(inv_desc)
    inv_objs = inv_objs.union(inv)

    inv = get_subtree(env.get_player_object().child, env.get_world_objects())
    surrounding = get_subtree(env.get_player_location().child, env.get_world_objects())
    player_obj = env.get_player_object()
    inv_attrs = []
    surr_attrs = []

    if player_obj in surrounding:
        surrounding.remove(player_obj)
    for i in inv:
        surrounding.remove(i)
        if gamename in readable:
            cur_attrs = attributes[gamename]
            obj_attrs = i.attr
            for oatr in obj_attrs:
                if oatr in cur_attrs.keys():
                    inv_attrs.append((i.name, cur_attrs[oatr].lower()))

    for s in surrounding:
        if gamename in readable:
            #print('read')
            cur_attrs = attributes[gamename]
            obj_attrs = s.attr
            for oatr in obj_attrs:
                if oatr in cur_attrs.keys():
                    surr_attrs.append((s.name, cur_attrs[oatr].lower()))
    surr_objs = surr_objs.union(' '.join([o.name for o in surrounding]).split())
    inv_objs = inv_objs.union(' '.join([o.name for o in inv]).split())

    # Filter out the objects that aren't in the dictionary
    def filter_words(objs):
        dict_words = [w.word for w in env.get_dictionary()]
        max_word_length = max([len(w) for w in dict_words])
        to_remove = set()
        for obj in objs:
            if obj[:max_word_length] not in dict_words:
                to_remove.add(obj)
        objs.difference_update(to_remove)
        return objs

    surr_objs = filter_words(surr_objs)
    inv_objs = filter_words(inv_objs)

    def filter_examinable(objs):
        desc2obj = {}
        # Filter out objs that aren't examinable
        for obj in objs:
            env.set_state(state)
            ex = clean(env.step('examine ' + obj)[0])
            if recognized(ex):
                if ex in desc2obj:
                    desc2obj[ex].append(obj)
                else:
                    desc2obj[ex] = [obj]
        env.set_state(state)
        return desc2obj

    surr_objs_final = filter_examinable(surr_objs)
    inv_objs_final = filter_examinable(inv_objs)

    surr_obj_names = []
    for s in surr_objs_final.values():
        surr_obj_names.extend(s)
    inv_obj_names = []
    for i in inv_objs_final.values():
        inv_obj_names.extend(i)

    #inv_attrs = set([(nm, attr) for (nm, attr) in inv_attrs if nm in inv_obj_names])
    inv_attrsd = {k: set() for k in inv_obj_names}
    for nm, attr in inv_attrs:
        nm = nm.split()
        for n in nm:
            if n in inv_obj_names:
                inv_attrsd[n].add(attr)

    #surr_attrs = set([(nm, attr) for (nm, attr) in surr_attrs if nm in surr_obj_names])
    surr_attrsd = {k: set() for k in surr_obj_names}
    for nm, attr in surr_attrs:
        nm = nm.split()
        for n in nm:
            if n in surr_obj_names:
                surr_attrsd[n].add(attr)

    inv_attrsd = {k: list(v) for k, v in inv_attrsd.items()}
    surr_attrsd = {k: list(v) for k, v in surr_attrsd.items()}

    return surr_objs_final, inv_objs_final, inv_attrsd, surr_attrsd


def get_objs(env):
    inv_objs = get_subtree(env.get_player_object().child, env.get_world_objects())
    surrounding = get_subtree(env.get_player_location().child, env.get_world_objects())
    player_obj = env.get_player_object()
    if player_obj in surrounding:
        surrounding.remove(player_obj)
    for inv_obj in inv_objs:
        surrounding.remove(inv_obj)
    for obj in inv_objs:
        env.step('examine ' + obj.name)
    json_inv_objs = [{'name':obj.name, 'num':obj.num} for obj in inv_objs]
    json_surr_objs = [{'name':obj.name, 'num':obj.num} for obj in surrounding]
    return json_inv_objs, json_surr_objs


def find_valid_actions(env, state, candidate_actions):
    if env.game_over() or env.victory() or env.emulator_halted():
        return []
    diff2acts = {}
    orig_score = env.get_score()
    for act in candidate_actions:
        env.set_state(state)
        if isinstance(act, defines.TemplateAction):
            obs, rew, done, info = env.step(act.action)
        else:
            obs, rew, done, info = env.step(act)
        if env.emulator_halted():
            # print('Warning: Environment halted.')
            env.reset()
            continue
        if info['score'] != orig_score or done or env.world_changed():
            # Heuristic to ignore actions with side-effect of taking items
            if '(Taken)' in obs:
                continue
            diff = str(env._get_world_diff())
            if diff in diff2acts:
                if act not in diff2acts[diff]:
                    diff2acts[diff].append(act)
            else:
                diff2acts[diff] = [act]
    valid_acts = {}
    for k,v in diff2acts.items():
        valid_acts[k] = max(v, key=verb_usage_count)
    env.set_state(state)
    return valid_acts


def load_bindings_updated(rom):
    rom = os.path.basename(rom)
    for k, v in defines_updated.BINDINGS_DICT.items():
        if k == rom or v['rom'] == rom:
            return v


def build_dataset_walkthrough(rom):
    gamename = rom.split('/')[1]

    bindings = load_bindings_updated(rom)
    env = FrotzEnv(rom, seed=bindings['seed'])
    obs = env.reset()[0]

    data = []
    prev_triples = []
    prev_act = ''
    prev_diff = ''

    prev_state = None
    curr_state = None
    try:
        walkthrough = bindings['walkthrough'].split('/')
        walkthrough = walkthrough[:int(len(walkthrough) / 3)]
    except KeyError:
        return data
    act_gen = TemplateActionGenerator(bindings)

    done = False
    for i, act in enumerate(walkthrough):
        print('w', i, rom)
        if done:
            break
        if i == 0:
            prev_score = env.get_score()
        score = env.get_score()
        state = env.get_state()
        data += build_dataset_from_state(deepcopy(rom), deepcopy(state))


        loc_desc = env.step('look')[0]
        env.set_state(state)
        inv_desc = env.step('inventory')[0]
        env.set_state(state)

        location = env.get_player_location()
        if location is None:
            break
        location_json = {'name':location.name, 'num': location.num}

        surrounding_objs, inv_objs, inv_attr, surr_attr = identify_interactive_objects(env, loc_desc, inv_desc, state, gamename)
        # inv_objs, surrounding_objs = get_objs(env)

        interactive_objs = [obj[0] for obj in env.identify_interactive_objects(use_object_tree=True)]
        candidate_actions = act_gen.generate_actions(interactive_objs)
        diff2acts = find_valid_actions(env, state, candidate_actions)

        obs_new, rew, done, info = env.step(act)
        # diff = str(env._get_world_diff())
        #if not str(diff) in diff2acts:
        #    print('WalkthroughAct: {} Diff: {} Obs: {}'.format(act, diff, clean(obs_new)))
        surrounding = get_subtree(env.get_player_location().child,
                                      env.get_world_objects())
        triples = tree_to_triple(env.get_player_location(),
                                 env.get_player_object(), surrounding, act,
                                 location, rom)
        diff = str(env._get_world_diff())

        #if prev_act != '':
        curr_state = {
            'walkthrough_act': prev_act,
            'walkthrough_diff': prev_diff,
            'obs': obs,
            'loc_desc': loc_desc,
            'inv_desc': inv_desc,
            'inv_objs': inv_objs,
            'inv_attrs': inv_attr,
            'location': location_json,
            'surrounding_objs': surrounding_objs,
            'surrounding_attrs': surr_attr,
            'graph': prev_triples,
            'valid_acts': diff2acts,
            'score': score
        }

        if prev_state is not None and curr_state is not None:
            triple_diff = graph_diff(prev_state['graph'], curr_state['graph'])
            data.append({
                'rom': bindings['name'],
                'state': prev_state,
                'next_state': curr_state,
                'graph_diff': triple_diff,
                'action': act,
                'reward': rew
            })

        prev_state = curr_state

        obs = obs_new
        prev_triples = triples
        prev_act = act
        prev_diff = diff

    env.close()

    return data


def build_dataset_from_state(rom, state):
    data = []


    gamename = rom.split('/')[1]
    visited = set()

    for i in range(1):
        # rom = './'roms/zork1.z5'
        bindings = load_bindings_updated(rom)
        env = FrotzEnv(rom, seed=bindings['seed'])
        env.set_state(state)
        obs = env.step('look')[0]#env.reset()[0]
        # walkthrough = bindings['walkthrough'].split('/')
        act_gen = TemplateActionGenerator(bindings)
        done = False
        prev_score = env.get_score()
        prev_triples = []
        prev_act = ''
        prev_diff = ''

        prev_state = None
        curr_state = None
        for j in range(3):
            #print(i, j, rom)
            if done:
                break
            score = env.get_score()
            state = env.get_state()

            loc_desc = env.step('look')[0]
            env.set_state(state)
            inv_desc = env.step('inventory')[0]
            env.set_state(state)

            #print(loc_desc, inv_desc)

            location = env.get_player_location()
            if location is None:
                break
            location_json = {'name': location.name, 'num': location.num}

            surrounding_objs, inv_objs, inv_attr, surr_attr = identify_interactive_objects(env, loc_desc, inv_desc, state, gamename)
            # inv_objs, surrounding_objs = get_objs(env)

            interactive_objs = [obj[0] for obj in env.identify_interactive_objects(use_object_tree=True)]
            candidate_actions = act_gen.generate_actions(interactive_objs)
            diff2acts = find_valid_actions(env, state, candidate_actions)

            valid_actions = list(diff2acts.values())
            # print(valid_actions)
            if len(valid_actions) == 0:
                done = True
                break
            act = random.choice(valid_actions)
            obs_new, rew, done, info = env.step(act)
            # diff = str(env._get_world_diff())
            #if not str(diff) in diff2acts:
            #    print('WalkthroughAct: {} Diff: {} Obs: {}'.format(act, diff, clean(obs_new)))
            surrounding = get_subtree(env.get_player_location().child,
                                          env.get_world_objects())
            triples = tree_to_triple(env.get_player_location(),
                                     env.get_player_object(), surrounding, act,
                                     location, rom)
            diff = str(env._get_world_diff())

            curr_state = {
                'walkthrough_act': prev_act,
                'walkthrough_diff': prev_diff,
                'obs': obs,
                'loc_desc': loc_desc,
                'inv_desc': inv_desc,
                'inv_objs': inv_objs,
                'inv_attrs': inv_attr,
                'location': location_json,
                'surrounding_objs': surrounding_objs,
                'surrounding_attrs': surr_attr,
                'graph': prev_triples,
                'valid_acts': diff2acts,
                'score': score
            }

            if prev_state is not None and curr_state is not None:
                triple_diff = graph_diff(prev_state['graph'],
                                         curr_state['graph'])
                data.append({
                    'rom': bindings['name'],
                    'state': prev_state,
                    'next_state': curr_state,
                    'graph_diff': triple_diff,
                    'action': act,
                    'reward': rew
                })

            prev_state = curr_state

            obs = obs_new
            prev_triples = triples
            prev_act = act
            prev_diff = diff

        env.close()


    return data


def build_dataset(rom):
    data = []

    gamename = rom.split('/')[1]
    visited = set()

    for i in range(1):
        # rom = './'roms/zork1.z5'
        bindings = load_bindings_updated(rom)
        env = FrotzEnv(rom, seed=bindings['seed'])
        obs = env.reset()[0]
        act_gen = TemplateActionGenerator(bindings)
        done = False
        prev_score = env.get_score()
        prev_triples = []
        prev_state = None
        curr_state = None
        prev_act = ''
        prev_diff = ''
        for j in range(100):
            if done:
                break
            score = env.get_score()
            state = env.get_state()


            loc_desc = env.step('look')[0]
            env.set_state(state)
            inv_desc = env.step('inventory')[0]
            env.set_state(state)

            #print(loc_desc, inv_desc)

            location = env.get_player_location()
            if location is None:
                break
            location_json = {'name': location.name, 'num': location.num}

            surrounding_objs, inv_objs, inv_attr, surr_attr = identify_interactive_objects(env, loc_desc, inv_desc, state, gamename)
            # inv_objs, surrounding_objs = get_objs(env)

            interactive_objs = [obj[0] for obj in env.identify_interactive_objects(use_object_tree=True)]
            candidate_actions = act_gen.generate_actions(interactive_objs)
            diff2acts = find_valid_actions(env, state, candidate_actions)

            valid_actions = list(diff2acts.values())
            if len(valid_actions) == 0:
                done = True
                break
            act = random.choice(valid_actions)
            obs_new, rew, done, info = env.step(act)

            surrounding = get_subtree(env.get_player_location().child,
                                          env.get_world_objects())
            triples = tree_to_triple(env.get_player_location(),
                                     env.get_player_object(), surrounding, act,
                                     location, rom)
            diff = str(env._get_world_diff())

            curr_state = {
                'walkthrough_act': prev_act,
                'walkthrough_diff': prev_diff,
                'obs': obs,
                'loc_desc': loc_desc,
                'inv_desc': inv_desc,
                'inv_objs': inv_objs,
                'inv_attrs': inv_attr,
                'location': location_json,
                'surrounding_objs': surrounding_objs,
                'surrounding_attrs': surr_attr,
                'graph': prev_triples,
                'valid_acts': diff2acts,
                'score': score
            }

            if prev_state is not None and curr_state is not None:
                triple_diff = graph_diff(prev_state['graph'],
                                         curr_state['graph'])
                data.append({
                    'rom': bindings['name'],
                    'state': prev_state,
                    'next_state': curr_state,
                    'graph_diff': triple_diff,
                    'action': act,
                    'reward': rew
                })

            prev_state = curr_state

            obs = obs_new
            prev_triples = triples
            prev_act = act
            prev_diff = diff
        env.close()


    return data

def build(game):
    try:
        print('start', game)
        load_attributes()
        gamename = game.split('/')[1].split('.')[0]
        if os.path.exists('data/data_' + gamename + '.json'):
            return
        data = []
        data_walk = build_dataset_walkthrough(game)
        data += data_walk

        with open('data/data_' + gamename + '.json', 'w') as f:
            json.dump(data, f)
    except RuntimeError:
        print('errored', game)
        return
    print(len(data))

    try:
        data_rand_walk = build_dataset(game)
        data += data_rand_walk

        with open('data/data_' + gamename + '.json', 'w') as f:
            json.dump(data, f)
    except RuntimeError:
        print('errored', game)
        return
    return




if __name__ == '__main__':
    from glob import glob

    load_attributes()

    games = glob('roms/*')

    for g in games:
        build(g)
