# Install deps
Mix.install([
  {:bumblebee, github: "elixir-nx/bumblebee", override: true},
  {:emlx, github: "elixir-nx/emlx"},
  #{:exla, "~> 0.9.2"},
  {:kino_bumblebee, "~> 0.5.1"},
  {:vix, "~> 0.31.1"}
])

defmodule ML do
  def test() do
    # Set defaults
    #EMLX
    Nx.global_default_backend({EMLX.Backend, device: :gpu})
    Nx.Defn.default_options(compiler: EMLX)

    #EXLA
    #Nx.global_default_backend(EXLA.Backend)
    
    # Load model
    model_name = "microsoft/resnet-18"
    {:ok, model} = Bumblebee.load_model({:hf, model_name})
    {:ok, featurizer} = Bumblebee.load_featurizer({:hf, model_name})
    serving = Bumblebee.Vision.image_classification(model, featurizer)
    
    {:ok, image} = Vix.Vips.Image.new_from_file("bird-original.jpg")
    {:ok, resized_image} = Vix.Vips.Operation.resize(image, 0.207)
    {:ok, %{data: data, shape: shape, type: type}} = Vix.Vips.Image.write_to_tensor(resized_image)
    
    image = data
      |> Nx.from_binary(:u8)
      |> Nx.reshape({224, 224, 3})
    
    Nx.Serving.run(serving, image)   
  end
end
