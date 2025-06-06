using Flux                           # Flux.jl v0.13+
using Flux: withgradient, update!
using Images
using Random, Statistics
using BSON: @save
using HuggingFaceDatasets            # for `load_dataset(...)`
using CUDA: CUDA                          # for GPU support

#-------------------------------------------
# 1) Check for GPU / Set DEVICE
#-------------------------------------------
DEVICE = CUDA.has_cuda() ? gpu : cpu
println("Using device: ", DEVICE === gpu ? "GPU" : "CPU")

#-------------------------------------------
# 2) Hyperparameters
#-------------------------------------------
struct HPARAMS
	image_size::Int       # 28 or 64
	out_channels::Int     # usually 3 (RGB)
	latent_dim::Int       # noise dimension (e.g. 100)
	batch_size::Int       # e.g. 128
	epochs::Int           # e.g. 50
	lr::Float32           # learning rate (2e-4)
	beta1::Float32        # Adam β₁ (0.5)
	keep_prob::Float32    # dropout keep prob (0.2)
end

hparams = HPARAMS(
	64,      # image_size (we’ll train on 64×64)
	3,       # out_channels (RGB)
	100,     # latent_dim (noise vector size)
	128,     # batch_size
	50,      # epochs
	1.0f-4,    # lr = 0.0001
	0.5f0,   # beta1 = 0.5
	0.2f0,    # dropout keep probability
)

#-------------------------------------------
# 3) DCGAN‐Style Weight Initialization
#-------------------------------------------
# Weight init ~ Normal(0, 0.02)
dcgan_winit = (dims...) -> 0.02f0 * randn(Float32, dims...)

# BatchNorm γ ∼ N(1, 0.02), β = 0
function init_batchnorm!(bn::BatchNorm)
	dims   = size(bn.γ)
	γ_cpu = 1.0f0 .+ 0.02f0 .* randn(Float32, dims)
	β_cpu = zeros(Float32, dims)
	bn.γ  .= DEVICE(γ_cpu)
	bn.β  .= DEVICE(β_cpu)
end

#-------------------------------------------
# 4) Data Loading via HuggingFaceDatasets
#-------------------------------------------
println("Loading CelebA‐faces from HuggingFace…")
ds = load_dataset("nielsr/CelebA-faces", split = "train").with_format("julia")
n_total = length(ds)
println("Total images in train split: $n_total")

# Limit how many images to use (optional—for speed/testing)
max_images = min(n_total, 160_000)

# 4.1) Preprocessing Function
#    - Input: record["image"] (RGB{N0f8} H×W×3)
#    - Output: Float32 tensor (H×W×3) ∈ [–1, +1]
function preprocess_hf(record, hp::HPARAMS)
	img_raw = record["image"]                         # Array{RGB{N0f8}, 2}
	arr_chw_n0f8 = channelview(img_raw)                   # (3, H, W), N0f8
	arr_chw = Float32.(arr_chw_n0f8) ./ 255            # (3, H, W), Float32 ∈ [0,1]
	img_hwc = permutedims(arr_chw, (2, 3, 1))          # (H, W, 3)

	# Resize to (hp.image_size × hp.image_size)
	img_rs = imresize(img_hwc, (hp.image_size, hp.image_size))  # (64,64,3)
	# Map [0,1] → [–1, +1]
	img_rs .= (img_rs .- 0.5f0) .* 2.0f0

	return img_rs  # (hp.image_size, hp.image_size, 3), Float32 ∈ [–1,+1]
end

println("Processing the first $max_images images into a 4D array…")
all_imgs = Array{Float32}(undef, hparams.image_size, hparams.image_size, hparams.out_channels, max_images)

for i in 1:max_images
	img_tensor = preprocess_hf(ds[i], hparams)
	all_imgs[:, :, :, i] = img_tensor
end

# Shuffle and partition into batches
# shuffle!(all_imgs, dims=4)
batches = [
	DEVICE(all_imgs[:, :, :, i:min(i+hparams.batch_size-1, max_images)])
	for i in 1:hparams.batch_size:max_images
]
println("Created $(length(batches)) batches of size ≤ $(hparams.batch_size).")

#-------------------------------------------
# 5) Generator & Discriminator Definitions
#-------------------------------------------

# 5.1) Generator: (64×64×3) output
function build_generator(hp::HPARAMS)
	return Chain(
		# 1) Project latent z ∈ ℝ^(latent_dim) → 4×4×1024
		Dense(hp.latent_dim, 4 * 4 * 1024; init = dcgan_winit),
		x -> reshape(x, (4, 4, 1024, :)),      # (4,4,1024,batch)
		BatchNorm(1024), x -> relu.(x),

		# 2) Upsample to 8×8
		ConvTranspose((4, 4), 1024 => 512; stride = 2, pad = 1, init = dcgan_winit),
		BatchNorm(512), x -> relu.(x),          # (8,8,512,batch)

		# 3) Upsample to 16×16
		ConvTranspose((4, 4), 512 => 256; stride = 2, pad = 1, init = dcgan_winit),
		BatchNorm(256), x -> relu.(x),          # (16,16,256,batch)

		# 4) Upsample to 32×32
		ConvTranspose((4, 4), 256 => 128; stride = 2, pad = 1, init = dcgan_winit),
		BatchNorm(128), x -> relu.(x),          # (32,32,128,batch)

		# 5) Upsample to 64×64
		ConvTranspose((4, 4), 128 => 64; stride = 2, pad = 1, init = dcgan_winit),
		BatchNorm(64), x -> relu.(x),           # (64,64,64,batch)

		# 6) Final conv → 64×64×3
		Conv((3, 3), 64 => hp.out_channels; pad = 1, init = dcgan_winit),
		x -> tanh.(x),                           # outputs in [–1, +1]
	)
end

# 5.2) Discriminator: (64×64×3) → 1 output
function build_discriminator(hp::HPARAMS)
	return Chain(
		# (1) 64×64×3 → 32×32×64
		Conv((5, 5), hp.out_channels => 64; stride = 2, pad = 2, init = dcgan_winit),
		x -> leakyrelu.(x, 0.2f0),
		Dropout(hp.keep_prob),

		# (2) 32×32×64 → 16×16×128
		Conv((5, 5), 64 => 128; stride = 2, pad = 2, init = dcgan_winit, bias = false),
		BatchNorm(128), x -> leakyrelu.(x, 0.2f0),
		Dropout(hp.keep_prob),

		# (3) 16×16×128 → 8×8×256
		Conv((5, 5), 128 => 256; stride = 2, pad = 2, init = dcgan_winit, bias = false),
		BatchNorm(256), x -> leakyrelu.(x, 0.2f0),
		Dropout(hp.keep_prob),

		# (4) 8×8×256 → 4×4×128
		Conv((5, 5), 256 => 128; stride = 2, pad = 2, init = dcgan_winit, bias = false),
		BatchNorm(128), x -> leakyrelu.(x, 0.2f0),
		Dropout(hp.keep_prob),

		# Flatten → Dense → Sigmoid
		x -> reshape(x, :, size(x, 4)),          # (4*4*128=2048, batch)
		Dense(4 * 4 * 128, 1; init = dcgan_winit),
		x -> σ.(x),                               # final probability
	)
end

generator     = DEVICE(build_generator(hparams))
discriminator = DEVICE(build_discriminator(hparams))

# Initialize BatchNorm layers
for layer in generator
	if layer isa BatchNorm
		init_batchnorm!(layer)
	end
end

for layer in discriminator
	if layer isa BatchNorm
		init_batchnorm!(layer)
	end
end

#-------------------------------------------
# 6) Loss & Optimizers
#-------------------------------------------
loss_bce(y_pred, y_true) = Flux.binarycrossentropy(y_pred, y_true)

opt_gen  = Flux.setup(Adam(hparams.lr, (hparams.beta1, 0.999f0)), generator)
opt_dscr = Flux.setup(Adam(hparams.lr, (hparams.beta1, 0.999f0)), discriminator)

# Put BatchNorm into train mode (per‐batch stats)
Flux.trainmode!(generator)
Flux.trainmode!(discriminator)

#-------------------------------------------
# 7) Training Steps
#-------------------------------------------
function train_discriminator!(gen::Chain, dsc::Chain, real_batch, opt_dscr, hp::HPARAMS)
	batch_size  = size(real_batch, 4)
	real_labels = 0.9f0 .+ (0.05f0 .* rand(Float32, 1, batch_size)) |> DEVICE  # [0.9, 0.95]
	fake_labels = 0.0f0 .+ (0.05f0 .* rand(Float32, 1, batch_size)) |> DEVICE  # [0.0, 0.05]

	d_loss, back = withgradient(dsc) do dsc
		# (A) D(real)
		pred_real = dsc(real_batch)
		loss_real = loss_bce(pred_real, real_labels)

		# (B) D(fake)
		noise     = randn(Float32, hp.latent_dim, batch_size) |> DEVICE
		fake_imgs = gen(noise)
		pred_fake = dsc(fake_imgs)
		loss_fake = loss_bce(pred_fake, fake_labels)

		return loss_real + loss_fake
	end

	Flux.update!(opt_dscr, dsc, back[1])
	return d_loss
end

function train_generator!(gen::Chain, dsc::Chain, real_batch, opt_gen, hp::HPARAMS)
	batch_size  = size(real_batch, 4)
	real_labels = ones(Float32, 1, batch_size) |> DEVICE

	g_loss, back = withgradient(gen) do gen
		noise     = randn(Float32, hp.latent_dim, batch_size) |> DEVICE
		fake_imgs = gen(noise)
		pred_fake = dsc(fake_imgs)
		return loss_bce(pred_fake, real_labels)
	end

	Flux.update!(opt_gen, gen, back[1])
	return g_loss
end

#-------------------------------------------
# 8) Sampling & Mosaic Utility
#-------------------------------------------
function sample_and_mosaic_conv(gen::Chain, hp::HPARAMS, Nrow::Int)
	Ngen = Nrow^2
	Z    = randn(Float32, hp.latent_dim, Ngen) |> DEVICE
	Xout = gen(Z) |> cpu

	imgs = Vector{Array{RGB{Float32}, 2}}(undef, Ngen)
	for i in 1:Ngen
		img = Xout[:, :, :, i]               # (64,64,3) ∈ [–1,+1]
		img .= (img .+ 1.0f0) ./ 2.0f0            # → [0,1]
		img_chw = permutedims(img, (3, 1, 2)) # (3,64,64)
		imgs[i] = colorview(RGB, img_chw)     # (64,64) Array{RGB,2}
	end

	return make_mosaic(imgs, Nrow)
end

#-------------------------------------------
# 9) Main Training Loop
#-------------------------------------------

function train()
	global_step = 0

	for epoch in 1:hparams.epochs
		println("Epoch $epoch / $(hparams.epochs)")
		for real_batch in batches
			# 1) Discriminator update
			d_loss = train_discriminator!(generator, discriminator, real_batch, opt_dscr, hparams)

			# 2) Generator update
			g_loss = train_generator!(generator, discriminator, real_batch, opt_gen, hparams)

			# 3) Logging
			if global_step % 100 == 0
				@info "Step $global_step:  D_loss=$(round(d_loss; digits=4)), G_loss=$(round(g_loss; digits=4))"
			end

			# 4) Sample & save every 500 steps
			if global_step % 500 == 0
				mosaic = sample_and_mosaic_conv(generator, hparams, 4)
				save("./output/dcgan_epoch$(epoch)_step$(global_step).png", mosaic)
				println("Saved sample at epoch $epoch, step $global_step")
			end

			global_step += 1
		end

		# 5) Save checkpoints at' end of epoch
		# @save "checkpoints/dcgan_gen_epoch$(epoch).bson" generator discriminator
		# println("Saved checkpoints for epoch $epoch")
	end
end

if abspath(PROGRAM_FILE) == @__FILE__
	train()
end
