import argparse
import enum

import openai

COMPLETION_MODEL = 'text-davinci-003'
CHAT_COMPLETION_MODEL = 'gpt-3.5-turbo'
QUIT_COMMAND = "#q"

input_is_consumed = False
prompt_is_consumed = False


class Conversation:
    __conversations = list()

    class Functionalities(enum.Enum):
        COMPLETION = 0,
        CHAT_COMPLETION = 1

    class Role(enum.Enum):
        SYSTEM = 0
        USER = 1
        ASSISTANT = 2

    class Message:
        def __init__(self, role, msg: str):
            self.role = role
            self.message = msg

        def __str__(self):
            return f'{self.role}: {self.message}'

    @staticmethod
    def push(role, msg: str):
        Conversation.__conversations.append(Conversation.Message(str(role).split(".")[-1].lower(), msg.strip()))

    @staticmethod
    def get_conversations():
        return Conversation.__conversations


def str2bool(_str):
    if isinstance(_str, bool):
        return _str
    if _str.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif _str.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_input():
    global input_is_consumed
    if args.option and not input_is_consumed:
        input_is_consumed = True
        return args.option.strip()
    else:
        input_is_consumed = True
        return input("Enter an option: ").strip()


def get_prompt():
    global prompt_is_consumed
    if args.prompt and not prompt_is_consumed:
        prompt_is_consumed = True
        return args.prompt.strip()
    else:
        prompt_is_consumed = True
        return input(f'Enter prompt ({QUIT_COMMAND} to quit): ').strip()


def list_models():
    _models = openai.Model.list()
    return _models['data'], len(_models['data'])


def split_logit_bias(logit_bias: str):
    dictionary = {}
    if logit_bias:
        array = logit_bias.strip().split(',')
        for item in array:
            items = item.strip().split(":")
            dictionary[items[0]] = int(items[1])

    return dictionary


def iterate_responses(responses, mode):
    _str = ""
    for response in responses:
        if mode == Conversation.Functionalities.CHAT_COMPLETION:
            for choice in response['choices']:
                if 'delta' in choice and 'content' in choice['delta']:
                    _str += choice['delta']['content']
                    yield _str.strip()
        elif mode == Conversation.Functionalities.COMPLETION:
            for choice in response['choices']:
                if 'text' in choice:
                    _str += choice['text']
                    yield _str.strip()


def handle_conversation(prompt, response, mode):
    Conversation.push(Conversation.Role.USER, prompt)
    if args.stream:
        complete_response = ""
        for resp in iterate_responses(response, mode):
            print(resp)
            complete_response = resp

        Conversation.push(Conversation.Role.ASSISTANT, complete_response)
    else:
        for choice in response['choices']:
            if mode == Conversation.Functionalities.CHAT_COMPLETION:
                Conversation.push(Conversation.Role.ASSISTANT, choice['message']['content'])
            elif mode == Conversation.Functionalities.COMPLETION:
                Conversation.push(Conversation.Role.ASSISTANT, choice['text'])
        print(response)


def chatCompletion():
    while True:
        print("\n-------------------------\n")
        prompt = get_prompt()
        if prompt == QUIT_COMMAND:
            break

        response = openai.ChatCompletion.create(
            model=CHAT_COMPLETION_MODEL,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            n=args.n if args.n is not None else 1,
            stream=args.stream,
            presence_penalty=args.presence_penalty,
            frequency_penalty=args.frequency_penalty,
            logit_bias=split_logit_bias(args.logit_bias),
            messages=[{'role': str(Conversation.Role.USER).split(".")[-1].lower(), 'content': prompt}]
        )

        handle_conversation(prompt, response, Conversation.Functionalities.CHAT_COMPLETION)


def completion():
    while True:
        print("\n-------------------------\n")
        prompt = get_prompt()
        if prompt == QUIT_COMMAND:
            break

        response = openai.Completion.create(
            model=COMPLETION_MODEL,
            suffix=args.suffix,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            n=args.n if args.n is not None else 1,
            best_of=args.best_of if args.best_of is not None else 1,
            stream=args.stream,
            logprobs=args.log_probs,
            echo=args.echo,
            stop=args.stop,
            presence_penalty=args.presence_penalty,
            frequency_penalty=args.frequency_penalty,
            logit_bias=split_logit_bias(args.logit_bias),
            prompt=prompt
        )

        handle_conversation(prompt, response, Conversation.Functionalities.COMPLETION)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--api_key", help="API key generated by OpenAI", type=str)
    parser.add_argument("--max_tokens", help="max_tokens parameter", type=int, default=5, nargs='?')
    parser.add_argument("--n", help="n parameter", type=int, default=None, nargs='?')
    parser.add_argument("--log_probs", help="logprobs parameter", type=int, default=None, nargs='?')
    parser.add_argument("--best_of", help="best_of parameter", type=int, default=None, nargs='?')
    parser.add_argument("--temperature", help="temperature parameter", type=float, default=1.0, nargs='?')
    parser.add_argument("--top_p", help="top_p parameter", type=float, default=1.0, nargs='?')
    parser.add_argument("--presence_penalty", help="presence_penalty parameter", type=float, default=0, nargs='?')
    parser.add_argument("--frequency_penalty", help="frequency_penalty parameter", type=float, default=0, nargs='?')
    parser.add_argument("--logit_bias", help="logit_bias parameter", type=str, default=None, nargs='?')
    parser.add_argument("--suffix", help="suffix parameter", type=str, default=None, nargs='?')
    parser.add_argument("--stop", help="stop parameter", type=str, default=None, nargs='?')
    parser.add_argument("--stream", help="stream parameter", type=str2bool, const=True, default=False, nargs='?')
    parser.add_argument("--echo", help="echo parameter", type=str2bool, const=True, default=False, nargs='?')
    parser.add_argument("--prompt", help="prompt to be used as the input command", default=None, nargs='?')
    parser.add_argument("--option", help="option to be used as the menu input value", default=None, nargs='?')
    args = parser.parse_args()

    if not args.api_key:
        print("Please provide api_key first")
        exit(0)
    if args.best_of is not None and args.n is not None and args.best_of <= args.n:
        print("best_of should be greater than n while using Completion")
        exit(0)

    openai.api_key = args.api_key

    print("\n============ BEGINNING ============\n")
    while True:
        print("1. Chat Completion")
        print("2. Completion")
        print("3. Models")
        print("4. Conversation history")
        print("5. Exit")
        print("")

        option = get_input()

        if not option.isdigit():
            print("Wrong input format!")
            continue

        option = int(option)

        if option == 1:
            chatCompletion()
        elif option == 2:
            completion()
        elif option == 3:
            models, size = list_models()
            print(f"{size} models were found! Here's the list: ")
            print(models)
        elif option == 4:
            print("Conversation history:")
            for message in Conversation.get_conversations():
                print(message)
        elif option == 5:
            exit(0)
        else:
            print("Wrong input!")
        print("")

    print("\n============ THE END ============\n")
