# JerichoWorld Dataset

Dataset corresponding to the paper Modeling Worlds in Text.

## Dataset Organization
* ```data/train.json```: The main dataset file containing a list of training examples.
* ```data/test.json```: The main dataset file containing a list of test examples.

The dataset is organized as:

```
{
"rom": "zork1",
"state": state,
"next_state": next_state,
"action": act,
"reward": rew,
}
```
Where `state` and `next_state` are two subsequent states with `act` being the corresponding action and `rew` being the corresponding reward. The rom names vary based on what is defined in the paper.

## Example of a state:
```json
{
    "walkthrough_act": "Open window",
    "walkthrough_diff": "((), ((235, 11),), ())",
    "obs": "Behind House\nYou are behind the white house. A path leads into the forest to the east. In one corner of the house there is a small window which is slightly ajar.\n\n",
    "loc_desc": "Behind House\nYou are behind the white house. A path leads into the forest to the east. In one corner of the house there is a small window which is slightly ajar.\n\n",
    "inv_desc": "You are carrying:\n  A jewel-encrusted egg\n\n",
    "inv_objs": {
      "The jewel encrusted egg is closed.": [
        "egg"
      ]
    },
    "location": {
      "name": "Behind House",
      "num": 79
    },
    "surrounding_objs": {
      "The window is slightly ajar, but not enough to allow entry.": [
        "small",
        "window"
      ],
      "The house is a beautiful colonial house which is painted white. It is clear that the owners must have been extremely wealthy.": [
        "white",
        "house"
      ],
      "There's nothing special about the way.": [
        "path"
      ]
    },
    "state": "saves/f461488f-3085-4f5a-ac2f-bd424561e8c6.pkl",
    "valid_acts": {
      "((), ((235, 11),), ())": "open small",
      "(((86, 4),), (), ())": "take on egg",
      "(((87, 79),), (), ())": "put down egg",
      "(((4, 80),), ((80, 3),), ())": "south",
      "(((87, 79), (86, 79)), (), ())": "throw egg at small",
      "(((4, 81),), (), ())": "north",
      "(((4, 74),), ((74, 3),), ())": "east"
    },
    "prev_graph": [
      [
        "you",
        "have",
        "jewel-encrusted egg"
      ],
      [
        "North House",
        "south",
        "Forest Path"
      ],
      [
        "you",
        "in",
        "North House"
      ],
      [
        "golden clockwork canary",
        "in",
        "jewel-encrusted egg"
      ]
    ],
    "graph": [
      [
        "you",
        "have",
        "jewel-encrusted egg"
      ],
      [
        "you",
        "in",
        "Behind House"
      ],
      [
        "Behind House",
        "east",
        "North House"
      ],
      [
        "golden clockwork canary",
        "in",
        "jewel-encrusted egg"
      ]
    ],
    "score": 5
  }
```

## Fields
Each example defines the following fields:
* **rom**: Name of the game that generated this example.
* **obs**: Narrative text returned by the game as a result of the last action.
* **loc_desc**: Text returned by *look* command from current location.
* **inv_desc**: Text returned by *inventory* command from current step.
* **inv_objs**: Dictionary of ```{obj_description : [obj_names]}``` containing detected objects in the player's inventory.
* **surrounding_objs**: Dictionary of ```{obj_description : [obj_names]}``` containing detected objects in the player's immediate surroundings.
* **score**: Current game score at this step.
* **location**: Name and number for the world-object corresponding to the player's current location.
* **state**: Path to pickle file containing saved game state.
* **walkthrough_act**: Action taken by the data collection agent from the current state.
* **walkthrough_diff**: ```world_diff``` corresponding to taking the taken action.
* **valid_acts**: list of ```world_diff : action_str```. The important part here is the world_diff since there are many action strings that can result in the same world diff.
* **graph**: list of lists of ```subject, relation, object``` of knowledge graph for current step

## Current Tasks
* **Predict knowledge graphs**: Given the current observations and graph can we predict some/all of the next knowledge graph? Ground truth given by in ```graph```.
* **Predict valid actions**: Given a the current observations and graph can we predict some/all of the valid actions? Ground truth given by world-diffs in ```valid_acts```.

## Acknowledgements
We thank https://github.com/microsoft/jericho and its creators for the walkthroughs of the games and the environments.
