# pylint: disable=C0103, C0116, C0301, C0114
# %%
import re

import pandas as pd

# %%
data = pd.read_parquet(
    "/home/user/sic-classification-utils/scripts/firestore_data/evaluation_df_with_sa_clean_codes.parquet"
)

# %%
column = "survey_assist_closed_question_option_"
selected_response = []

# iterate through all rows of the data
for row in range(len(data)):
    # save the response selected by the user
    response = data["survey_assist_closed_question_response"][row]

    # check only when closed question was asked
    if response is not None:

        # change options from closed questions to lower case (matching the selected response)
        for i in range(1, 7):
            column_name = column + str(i)
            if data[column_name][row] is not None:
                data.loc[row, column_name] = data[column_name][row].lower()

        # find the order of the selected response
        j = 1
        while j < 7:  # noqa: PLR2004
            column_name = column + str(j)
            if response == data[column_name][row]:
                selected_response.append(j)
                j = 7
            j += 1

# %%
# percentage of codes found using closed quesiton
print(round(100 * len(selected_response) / len(data), 2))

# %%
selected_response_order = {}
for i in range(1, 7):
    picked_response = "option_" + str(i)
    selected_response_order[picked_response] = selected_response.count(i)

# %%
# order, which answer was seleceted
print(selected_response_order)
# order of the question selected
print(
    "1st option [%]:",
    round(100 * selected_response_order["option_1"] / len(selected_response), 2),
)
print(
    "2nd option [%]:",
    round(100 * selected_response_order["option_2"] / len(selected_response), 2),
)
print(
    "3rd option [%]:",
    round(100 * selected_response_order["option_3"] / len(selected_response), 2),
)
print(
    "4th option [%]:",
    round(100 * selected_response_order["option_4"] / len(selected_response), 2),
)
print(
    "5th option [%]:",
    round(100 * selected_response_order["option_5"] / len(selected_response), 2),
)
print(
    "6th option [%]:",
    round(100 * selected_response_order["option_6"] / len(selected_response), 2),
)

# %%
# sic_knowledge = pd.read_csv("/home/user/sic-classification-utils/notebooks/sic_data/sic_knowledge_base_utf8.csv")
sic_rephrased = pd.read_csv(
    "/home/user/sic-classification-utils/notebooks/sic_data/sic_rephrased_descriptions_2025_02_03.csv"
)


# %%
def convert_to_dict(dict_string):
    clean_string = dict_string.strip().strip("'")

    code = re.search(r"{Code: \s*(.*?),\s*Title: ", clean_string, re.DOTALL)
    code_value = code.group(1).strip() if code else ""

    title = re.search(
        r"Title: \s*(.*?),\s*Example activities: ", clean_string, re.DOTALL
    )
    title_value = title.group(1).strip() if title else ""
    title_value = title_value.lower()

    activities = re.search(r"Example activities: \s*(.*?)\s*}", clean_string, re.DOTALL)
    activities_value = activities.group(1).strip() if activities else ""

    result_dictionary = {
        "Code": code_value,
        "Title": title_value,
        "Example activities": activities_value,
    }
    return result_dictionary


# %%
# sic_knowledge['description'] = sic_knowledge['description'].str.lower()
sic_rephrased["reviewed_description"] = sic_rephrased[
    "reviewed_description"
].str.lower()
sic_rephrased["llm_rephrased_description"] = sic_rephrased[
    "llm_rephrased_description"
].str.lower()


# %%
def get_code_by_title(dictionary, title):
    for item in dictionary:
        if item.get("Title") == title:
            return item.get("Code")
    return "XXXXX"


# %%
sic_dictionary = sic_rephrased["input_description"].apply(convert_to_dict)
response_codes = []
none_of_the_above = 0

for row in range(len(data)):
    # save the response selected by the user
    if data["survey_assist_closed_question_response"][row] is not None:
        response = data["survey_assist_closed_question_response"][row].lower()

        if response == "none of the above":
            none_of_the_above += 1
            response_codes.append("None")

        elif response in list(sic_rephrased["reviewed_description"]):
            sic_code = sic_rephrased[sic_rephrased["reviewed_description"] == response][
                "input_code"
            ].item()
            response_codes.append(sic_code)

        # elif response in list(sic_knowledge['description']):
        #     sic_code = sic_knowledge[sic_knowledge['description'] == response]['label'].item()
        #     response_codes.append(sic_code)

        else:
            sic_code = get_code_by_title(sic_dictionary, response)
            response_codes.append(sic_code)

    else:
        response_codes.append("None")


# %%
# we get "None" when the closed question was not asked or the answer was "none of the above"
response_codes.count("None")

# %%
response_codes.count("XXXXX")

# %%
len(response_codes)

# %%
# asked closed question, but didn't get a code
round(100 * none_of_the_above / len(selected_response), 2)

# %%
data["survey_assist_closed_question_response_code"] = response_codes
