import onnxruntime_genai as og
import json
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Run multimodal inference with ONNX Runtime GenAI")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model directory")
    parser.add_argument("--image_paths", type=str, nargs="+", required=True, help="Path(s) to image file(s)")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt for the model")
    return parser.parse_args()

def main():
    args = parse_args()

    model_path = args.model_path
    image_paths = args.image_paths
    prompt = args.prompt

    config = og.Config(model_path)
    model = og.Model(config)
    tokenizer = og.Tokenizer(model)
    processor = model.create_multimodal_processor()
    stream = processor.create_stream()

    if len(image_paths) != 0:
        images = og.Images.open(*image_paths)
    else:
        images = None
    
    messages = []
    content_list = [{"type": "image"} for _ in image_paths]
    content_list.append({"type": "text", "text": prompt})
    messages.append({"role": "user", "content": content_list})

    message_json = json.dumps(messages)
    prompt = tokenizer.apply_chat_template(message_json, add_generation_prompt=True)
    inputs = processor(prompt, images=images)

    params = og.GeneratorParams(model)
    params.set_search_options(max_length=8192)
    generator = og.Generator(model, params)
    generator.set_inputs(inputs)

    while not generator.is_done():
        generator.generate_next_token()
        new_token = generator.get_next_tokens()[0]
        print(stream.decode(new_token), end="", flush=True)

    print()  # newline after generation

if __name__ == "__main__":
    main()