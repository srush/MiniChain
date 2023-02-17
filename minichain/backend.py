
class Google:
    def __init__(self, serpapi_key):
        from serpapi import GoogleSearch
        self.serpapi_key = serpapi_key



        
    def run(self, question):
        params = {
            "api_key": self.serpapi_key,
            "engine": "google",
            "q": question,
            "google_domain": "google.com",
            "gl": "us",
            "hl": "en"
        }


        with io.capture_output() as captured: #disables prints from GoogleSearch
            search = GoogleSearch(params)
            res = search.get_dict()

        if 'answer_box' in res.keys() and 'answer' in res['answer_box'].keys():
            toret = res['answer_box']['answer']
        elif 'answer_box' in res.keys() and 'snippet' in res['answer_box'].keys():
            toret = res['answer_box']['snippet']
        elif 'answer_box' in res.keys() and 'snippet_highlighted_words' in res['answer_box'].keys():
            toret = res['answer_box']["snippet_highlighted_words"][0]
        elif 'snippet' in res["organic_results"][0].keys():
            toret= res["organic_results"][0]['snippet'] 
        else:
            toret = None
        return toret


class OpenAI:
    def __init__(self, api_key):
        import openai
        openai = api_key
        
    def run(self, question):
        ans = openai.Completion.create(
            model="text-davinci-002",
            max_tokens=256,
            stop=stop,
            prompt=cur_prompt,
            temperature=0)
        return ans['choices'][0]['text']
