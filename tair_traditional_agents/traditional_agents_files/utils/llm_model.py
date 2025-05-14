import openai
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.models.mixtral import MixtralForCausalLM

import torch

class Llm_model:
    def __init__(self):
        openai.api_key = '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'
        self.first_comment = "You should now act as a mature driving assistant. I will provide you with the names of objects, their polar coordinates relative to the ego vehicle, and their speeds. These objects may include the ego vehicle itself, other vehicles, a waypoint to follow, and traffic lights. Each vehicle and pedestrian has its own ID number. Analyze the scene step-by-step to ensure safe driving and adherence to the designated route's waypoints."
        self.input_comment = "Information about current scene as follows; "
        self.security_comment = "Is there any vehicle that the ego vehicle might hit ?"
        self.second_comment = "Summarize the current scene without using any id number with two sentences."
        self.third_comment = "Then, what action should the ego vehicle take to reach the destination point? You need to choose an action. You can only choose one of these three actions: 'current lane,' 'left lane,' or 'right lane.' If you choose 'current lane,' the ego vehicle will keep to its current lane. If you choose 'left lane,' the ego vehicle will change to the left lane. If you choose 'right lane,' the ego vehicle will change to the right lane."
        self.fourth_comment = "Then, Inform me about the action that should be taken by the ego vehicle."
        self.action_comment = "Therefore, which one do you choose: left lane, current lane, or right lane?. Answer with two words."
        self.fifth_comment = "Explain the reason of the action with two sentences. "
        self.send_count = 0
        self.send_threshold = 50

        #self.moe_model = MixtralForCausalLM(moe_config,create_model=False,model=_model.model,lm_head=_model.lm_head)

        model_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"

        #self.moe_tokenizer = AutoTokenizer.from_pretrained(model_id)

        #self.moe_model = AutoModelForCausalLM.from_pretrained(model_id, load_in_8bit=True)

        #with open('/workspace/tg22/moe_model.pkl', 'rb') as file: #model.save_pretrained("/path/to/your/directory")
        #    self.moe_model = pickle.load(file)  # tugrul !!!!
        #self.moe_model.run_post_init()
        asd = 0
        self.high_level_action = 'current_lane'


    def create_message(self,scene_info):
        scene_messages = ""
        object_message_list = []
        for obj in scene_info:
            if obj['class'] == 'Ego':
                scene_messages += obj['class'] + " has " + str(obj['yaw']) + " angle and " + str(obj['speed']) + " m/s speed."
            elif obj['class'] == 'Traffic light':
                scene_messages += obj['class'] + " is " + str(obj['state']) + " state and its distance is "+ str(obj['distance']) + '.'
            elif obj['class'] == "Route":
                scene_messages += "Destination point's distance " + " is " + str(obj['distance']) + " and " + "the angle from the ego vehicle to the destination point is " +str(obj['yaw']) + " degree."
            elif obj['class'] == "Lane":
                if obj['right_lane']:
                    scene_messages += "There is a right lane."
                else:
                    scene_messages += "There is not a right lane."

                if obj['left_lane']:
                    scene_messages += "There is a left lane."
                else:
                    scene_messages += "There is not a left lane."

            elif obj['class'] == "Radar":
                if obj['obstacle']:
                    scene_messages += "According to the radar sensor,there is an object in the ego vehicle's lane. It might be a car if there is a car infront of the ego vehicle. You can consider to change the lane if there is no traffic sign and an object infront of the ego vehicle is stationary. "
            else:
                scene_messages += "a new "+ obj['class']+"'s color is "+ str(obj['color'])+ '. ' + obj['class']+"'s distance " + " is "+ str(obj['distance']) + " and this "+ obj['class']+"'s speed " + " is "+ str(obj['speed']) + " and "+ "the angle from the ego vehicle to the"+ obj['class'] +" is " +str(obj['yaw']) + " degree."

                object_message = obj["color"]+"'s distance " + " is "+ str(obj['distance']) + " speed " + " is "+ str(obj['speed']) + " and "+ "the angle"  +str(obj['yaw']) + " degree."

                object_message_list.append(object_message)

        messages = [
            self.first_comment + self.input_comment + scene_messages,
            self.security_comment,
            self.second_comment,
            self.third_comment,
            self.fourth_comment,
            self.action_comment,
            self.fifth_comment,
        ]

        return messages


    def send_message_to_gpt(self, message_list):
        reply_list = []
        messages = []
        for index, message in enumerate(message_list):
            if message:
                messages.append(
                    {"role": "user", "content": message},
                )
                #chat = openai.ChatCompletion.create(
                #    model="gpt-3.5-turbo", messages=messages
                #)

            #reply = chat.choices[0].message.content#"reply"+str(index)#
            if index >=2:
                reply_list.append(reply)
            #print(f"ChatGPT: {reply}")
            messages.append({"role": "assistant", "content": reply})
        return reply_list

    def __call__(self, scene_info):
        reply_list = []

        if len(scene_info) != 0 and self.send_count%self.send_threshold == 0:
            messages = self.create_message(scene_info)
            #reply_list = self.send_message_to_moe(messages)

        self.send_count += 1

        if len(reply_list) == 0:
            reply_list.append("no reply")
        else:
            self.high_level_action = reply_list[1]

        return reply_list, self.high_level_action
    
    def send_message_to_moe(self,message_list=None):

        reply_list = []
        reply = ""
        for index, message in enumerate(message_list):
            new_message = reply + message
            inputs = self.moe_tokenizer(new_message, return_tensors="pt")

            with torch.no_grad():
                outputs = self.moe_model.generate(**inputs, max_new_tokens=300)
            reply = self.moe_tokenizer.decode(outputs[0], skip_special_tokens=True)
            new_reply = ''.join(list(reply)[len(list(new_message)) + 2:])

            if index in [2, 5, 6]:
                reply_list.append(new_reply)


        return reply_list