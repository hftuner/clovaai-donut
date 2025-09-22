class DataProcessor:
    def __init__(self):
        self.new_special_tokens = set()
    
    def clear_new_special_tokens(self):
        self.new_special_tokens = set()
    
    def get_new_special_tokens(self):
        return list(self.new_special_tokens)

    def json2token(self, obj, update_special_tokens_for_json_key: bool = False, sort_json_key: bool = True):
        """
        Convert an ordered JSON object into a token sequence
        """
        
        if type(obj) == dict:
            if len(obj) == 1 and "text_sequence" in obj:
                return obj["text_sequence"]
            else:
                output = ""
                if sort_json_key:
                    keys = sorted(obj.keys(), reverse=True)
                else:
                    keys = obj.keys()
                for k in keys:
                    if update_special_tokens_for_json_key:
                        self.new_special_tokens.add(fr"<s_{k}>")
                        self.new_special_tokens.add(fr"</s_{k}>")
                    # xml style string
                    json_string = self.json2token(obj[k],update_special_tokens_for_json_key, sort_json_key)
                    output += ( fr"<s_{k}>" + json_string + fr"</s_{k}>")
                return output
        
        elif type(obj) == list:
            json_strings = []
            for item in obj:
                json_string = self.json2token(item, update_special_tokens_for_json_key, sort_json_key)
                json_strings.append(json_string)
            output = r"<sep/>".join(json_strings)
            return output
        
        else:
            return str(obj)