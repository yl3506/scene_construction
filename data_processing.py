import pandas as pd
import numpy as np

def load_data(file_path='/Users/yichen/Downloads/scene_construction/version6_pilot1_cleaned.csv'):
    # Load your data from a CSV file
    data = pd.read_csv(file_path)
    return data

def preprocess_data(data):
    # Add ParticipantID by numbering the rows
    data = data.reset_index(drop=True)
    data['ParticipantID'] = data.index + 1  # Start IDs from 1

    # Exclude invalid participants
    if 'invalid' in data.columns:
        data = data[data['invalid'] != 1]

    # Map 'condition' to scaling factors
    scaling_factor_mapping = {'250ms': 1, '500ms': 2, '1000ms': 4, '2000ms': 8}
    data['ScalingFactor'] = data['condition'].map(scaling_factor_mapping)

    # Set Default number of nodes that can be expanded (scales with duration allowed)
    base_time_limit = 1.0

    # Effective Time Limit will be calculated during model fitting using the parameters

    # Prepare properties per scene
    scene_properties_mapping = {
    # scene: 
    #   {property name: (column for yes-property, column for yes-rank, column for no-property, short name for this property)}
        'Drop':{
          'The location of the dropped thing relative to the other thing (e.g. to the left/right/front/back)':
              ('DropQ_0_GROUP_1', 'DropQ_0_1_RANK', 'DropQ_1_GROUP_1', 'loc(s1, s2)'),
          'The size of the dropped thing':
              ('DropQ_0_GROUP_2', 'DropQ_0_2_RANK', 'DropQ_1_GROUP_2', 'size(s1)'),
          'The trajectory of the dropped thing':
              ('DropQ_0_GROUP_3', 'DropQ_0_3_RANK', 'DropQ_1_GROUP_3', 'traj(s1)'),
          'A person who dropped the thing':
              ('DropQ_0_GROUP_4', 'DropQ_0_4_RANK', 'DropQ_1_GROUP_4', 'person'),
          'What was the thing being dropped (e.g. what object was it?)':
              ('DropQ_0_GROUP_5', 'DropQ_0_5_RANK', 'DropQ_1_GROUP_5', 'type(s1)'),
          'What was the other thing (e.g. what object was it?)':
              ('DropQ_0_GROUP_6', 'DropQ_0_6_RANK', 'DropQ_1_GROUP_6', 'type(s2)'),
          'The color of the dropped thing':
              ('DropQ_0_GROUP_7', 'DropQ_0_7_RANK', 'DropQ_1_GROUP_7', 'color(s1)'),
          'The color of the other thing':
              ('DropQ_0_GROUP_8', 'DropQ_0_8_RANK', 'DropQ_1_GROUP_8', 'color(s2)'),
          'The apparent material of the dropped thing':
              ('DropQ_0_GROUP_9', 'DropQ_0_9_RANK', 'DropQ_1_GROUP_9', 'material(s1)'),
          },

        'Push':
          {
          'The location of the pushed thing relative to the other thing (e.g. to the left/right/front/back)':
              ('PushQ_0_GROUP_1', 'PushQ_0_1_RANK', 'PushQ_1_GROUP_1', 'loc(s1, s2)'),
          'The weight of the pushed thing':
              ('PushQ_0_GROUP_2', 'PushQ_0_2_RANK', 'PushQ_1_GROUP_2', 'weight(s1)'),
          'The direction of movement of the pushed thing':
              ('PushQ_0_GROUP_3', 'PushQ_0_3_RANK', 'PushQ_1_GROUP_3', 'direction(s1)'),
          'A surface/table/floor':
              ('PushQ_0_GROUP_4', 'PushQ_0_4_RANK', 'PushQ_1_GROUP_4', 'surface'),
          'What was the thing being pushed (e.g. what object was it?)':
              ('PushQ_0_GROUP_5', 'PushQ_0_5_RANK', 'PushQ_1_GROUP_5', 'type(s1)'),
          'What was the other thing (e.g. what object was it?)':
              ('PushQ_0_GROUP_6', 'PushQ_0_6_RANK', 'PushQ_1_GROUP_6', 'type(s2)'),
          'The color of the pushed thing':
              ('PushQ_0_GROUP_7', 'PushQ_0_7_RANK', 'PushQ_1_GROUP_7', 'color(s1)'),
          'The color of the other thing':
              ('PushQ_0_GROUP_8', 'PushQ_0_8_RANK', 'PushQ_1_GROUP_8', 'color(s2)'),
          'The apparent material of the pushed thing':
              ('PushQ_0_GROUP_9', 'PushQ_0_9_RANK', 'PushQ_1_GROUP_9', 'material(s1)'),
          },

        'Pull':
          {
          'The location of the pulled thing relative to the other thing (e.g. to the left/right/front/back)':
              ('PullQ_0_GROUP_1', 'PullQ_0_1_RANK', 'PullQ_1_GROUP_1', 'loc(s1, s2)'),
          'The size of the pulled thing':
              ('PullQ_0_GROUP_2', 'PullQ_0_2_RANK', 'PullQ_1_GROUP_2', 'size(s1)'),
          'The trajectory of the pulled thing':
              ('PullQ_0_GROUP_3', 'PullQ_0_3_RANK', 'PullQ_1_GROUP_3', 'traj(s1)'),
          'A person who pulled the thing':
              ('PullQ_0_GROUP_4', 'PullQ_0_4_RANK', 'PullQ_1_GROUP_4', 'person'),
          'The type of the thing being pulled (e.g. what object was it?)':
              ('PullQ_0_GROUP_5', 'PullQ_0_5_RANK', 'PullQ_1_GROUP_5', 'type(s1)'),
          'The type of the other thing (e.g. what object was it?)':
              ('PullQ_0_GROUP_6', 'PullQ_0_6_RANK', 'PullQ_1_GROUP_6', 'type(s2)'),
          'The color of the pulled thing':
              ('PullQ_0_GROUP_7', 'PullQ_0_7_RANK', 'PullQ_1_GROUP_7', 'color(s1)'),
          'The color of the other thing':
              ('PullQ_0_GROUP_8', 'PullQ_0_8_RANK', 'PullQ_1_GROUP_8', 'color(s2)'),
          'The apparent material of the pulled thing':
              ('PullQ_0_GROUP_9', 'PullQ_0_9_RANK', 'PullQ_1_GROUP_9', 'material(s1)'),
          },
        }

    # Initialize a list to hold processed participant data
    processed_data_list = []

    # Iterate over participants
    for idx, row in data.iterrows():
        participant_id = row['ParticipantID']
        scene = row['scene']
        scaling_factor = row['ScalingFactor']

        # Get properties mapping for the participant's scene
        properties_mapping = scene_properties_mapping[scene]

        # Initialize dictionaries for responses and ranks
        responses = {}
        ranks = {}
        property_short_names = []

        for prop_desc, (yes_col, rank_col, no_col, prop_short_name) in properties_mapping.items():
            # Check if participant responded 'yes' or 'no'
            yes_response = row.get(yes_col)
            no_response = row.get(no_col)

            if not pd.isnull(yes_response) and yes_response != '':
                # Participant said 'yes'
                responses[prop_short_name] = 1
                rank = row.get(rank_col)
                if pd.isnull(rank):
                    rank = np.nan  # Handle missing rank
                ranks[prop_short_name] = rank
            elif not pd.isnull(no_response) and no_response != '':
                # Participant said 'no'
                responses[prop_short_name] = 0
                ranks[prop_short_name] = np.nan  # No rank for 'no' responses
            else:
                # Missing data
                responses[prop_short_name] = np.nan
                ranks[prop_short_name] = np.nan

            property_short_names.append(prop_short_name)

        # Create participant data dictionary
        participant_data = {
            'ParticipantID': participant_id,
            'Scene': scene,
            'ScalingFactor': scaling_factor,
            'BaseTimeLimit': base_time_limit,
            'Properties': property_short_names,
        }

        # Add responses and ranks to participant data
        for prop in property_short_names:
            participant_data[f'Response_{prop}'] = responses[prop]
            participant_data[f'Rank_{prop}'] = ranks[prop]

        processed_data_list.append(participant_data)

    # Convert to DataFrame
    processed_data = pd.DataFrame(processed_data_list)

    return processed_data
