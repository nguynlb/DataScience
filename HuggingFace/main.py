from langchain.llms import OpenAI
from dotenv import load_dotenv

load_dotenv()

def generate_pet_name():
    llm = OpenAI(temperature=.7)

    name = llm("I have a dot pet and I want a cool name for him. Can you give me some cool names for my pet?")

    return name

if __name__ == "__main__":
    print(generate_pet_name())
