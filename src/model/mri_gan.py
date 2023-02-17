from pathlib import Path
import tensorflow as tf
from matplotlib import pyplot as plt
from IPython import display
import time
import tensorflow.keras.backend as K

from src.model.generator import Generator
from src.model.discriminator import Discriminator

class MRIGan():

    def __init__(self, dataset_pipeline, epochs, checkpoint_path="./checkpoint") -> None:
        self.generator_t1_to_t2 = Generator()
        self.generator_t2_to_t1 = Generator()

        self.discriminator_t1 = Discriminator()
        self.discriminator_t2 = Discriminator()
        
        self.dataset_pipeline = dataset_pipeline

        self.example_t1, self.example_t2 = next(iter(self.dataset_pipeline))
        self.example_t1 = self.example_t1[0:4]
        self.example_t2 = self.example_t2[0:4]

        self.cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

        self.generator_t1_to_t2_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        self.generator_t2_to_t1_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

        self.discriminator_t1_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        self.discriminator_t2_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

        self.checkpoint = tf.train.Checkpoint(
            generator_t1_to_t2_optimizer=self.generator_t1_to_t2_optimizer,
            generator_t2_to_t1_optimizer=self.generator_t2_to_t1_optimizer,
            discriminator_t1_optimizer=self.discriminator_t1_optimizer,
            discriminator_t2_optimizer=self.discriminator_t2_optimizer,
            generator_t1_to_t2=self.generator_t1_to_t2,
            generator_t2_to_t1=self.generator_t2_to_t1,
            discriminator_t1=self.discriminator_t1,
            discriminator_t2=self.discriminator_t2
        )

        self.checkpoint_manager = tf.train.CheckpointManager(
            self.checkpoint, checkpoint_path, max_to_keep=5
        )

        self.epochs = epochs
        self.current_epoch = 0


    def generator_loss(self, fake_output):
        return self.cross_entropy(tf.ones_like(fake_output), fake_output)
    
    def discriminator_loss(self, real_output, fake_output):
        real_loss = self.cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = self.cross_entropy(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss
        return total_loss * 0.5
    
    def cycle_loss(self, real_image, cycled_image):
        loss = tf.reduce_mean(tf.abs(real_image - cycled_image))
        return 10.0 * loss
    
    def identity_loss(self, real_image, same_image):
        loss = tf.reduce_mean(tf.abs(real_image - same_image))
        return 0.5 * loss
    
    def init(self):
        self.checkpoint.restore(self.checkpoint_manager.latest_checkpoint)
        if self.checkpoint_manager.latest_checkpoint:
            print("Restored from {}".format(self.checkpoint_manager.latest_checkpoint))
        else:
            print("Initializing from scratch.")


    def plot_images(self, clear_output=True, save_path="./images"):

        if not Path(save_path).exists():
            Path(save_path).mkdir(parents=True, exist_ok=True)

        if clear_output:
            display.clear_output(wait=True)

        # plot example t1 only in first row and predict t2 in second row, no need to plot t2
        prediction_t1_to_t2 = self.generator_t1_to_t2(self.example_t1, training=False)

        plt.figure(figsize=(10,10))
        for i in range(4):
            plt.subplot(4, 2, 2*i+1)
            plt.imshow(self.example_t1[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
            plt.axis('off')

            plt.subplot(4, 2, 2*i+2)
            plt.imshow(prediction_t1_to_t2[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
            plt.axis('off')
        
        plt.savefig(
            Path(save_path) / f"epoch_{self.current_epoch}.png"
        )

        plt.show()

    @tf.function
    def train_step(self, t1, t2):
        
        with tf.GradientTape(persistent=True) as tape:
            fake_y = self.generator_t1_to_t2(t1, training=True)
            cycled_x = self.generator_t2_to_t1(fake_y, training=True)

            fake_x = self.generator_t2_to_t1(t2, training=True)
            cycled_y = self.generator_t1_to_t2(fake_x, training=True)

            same_x = self.generator_t2_to_t1(t1, training=True)
            same_y = self.generator_t1_to_t2(t2, training=True)

            disc_real_x = self.discriminator_t1(t1, training=True)
            disc_real_y = self.discriminator_t2(t2, training=True)

            disc_fake_x = self.discriminator_t1(fake_x, training=True)
            disc_fake_y = self.discriminator_t2(fake_y, training=True)

            # calculate the loss
            gen_t1_to_t2_loss = self.generator_loss(disc_fake_y)
            gen_t2_to_t1_loss = self.generator_loss(disc_fake_x)

            total_cycle_loss = self.cycle_loss(t1, cycled_x) + self.cycle_loss(t2, cycled_y)

            # Total generator loss = adversarial loss + cycle loss
            total_gen_t1_to_t2_loss = gen_t1_to_t2_loss + total_cycle_loss + self.identity_loss(t2, same_y)
            total_gen_t2_to_t1_loss = gen_t2_to_t1_loss + total_cycle_loss + self.identity_loss(t1, same_x)

            disc_t1_loss = self.discriminator_loss(disc_real_x, disc_fake_x)
            disc_t2_loss = self.discriminator_loss(disc_real_y, disc_fake_y)

        # Calculate the gradients for generator and discriminator
        generator_t1_to_t2_gradients = tape.gradient(total_gen_t1_to_t2_loss,
                                                    self.generator_t1_to_t2.trainable_variables)
        generator_t2_to_t1_gradients = tape.gradient(total_gen_t2_to_t1_loss,
                                                    self.generator_t2_to_t1.trainable_variables)
        
        discriminator_t1_gradients = tape.gradient(disc_t1_loss,
                                                self.discriminator_t1.trainable_variables)
        discriminator_t2_gradients = tape.gradient(disc_t2_loss,
                                                self.discriminator_t2.trainable_variables)

        # Apply gradients
        self.generator_t1_to_t2_optimizer.apply_gradients(
            zip(generator_t1_to_t2_gradients, self.generator_t1_to_t2.trainable_variables)
        )

        self.generator_t2_to_t1_optimizer.apply_gradients(
            zip(generator_t2_to_t1_gradients, self.generator_t2_to_t1.trainable_variables)
        )

        self.discriminator_t1_optimizer.apply_gradients(
            zip(discriminator_t1_gradients, self.discriminator_t1.trainable_variables)
        )

        self.discriminator_t2_optimizer.apply_gradients(
            zip(discriminator_t2_gradients, self.discriminator_t2.trainable_variables)
        )

    
    def train(self):
        for epoch in range(self.current_epoch, self.epochs):
            start = time.time()
            print(f"Epoch: {epoch}")
            # Train
            for n, (t1, t2) in enumerate(self.dataset_pipeline):
                losses = self.train_step(t1, t2)
                print(f"\r{n+1}/{len(self.dataset_pipeline)}", end="")

            # Plot
            self.plot_images(clear_output=True)
            # self.plot_losses(clear_output=False)

            # Save model
            self.checkpoint_manager.save()

            print(f" - time: {time.time()-start}")

            self.current_epoch += 1
    
