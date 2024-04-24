package net.mlcraft.elevationmod;

import com.mojang.serialization.Codec;
import net.fabricmc.api.ModInitializer;

import net.minecraft.registry.*;
import net.minecraft.util.Identifier;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;

public class MLCraft implements ModInitializer {
	public static final String MOD_ID = "mlcraft";
    public static final Logger LOGGER = LoggerFactory.getLogger(MOD_ID);

	@Override
	public void onInitialize() {
		// This code runs as soon as Minecraft is in a mod-load-ready state.
		// However, some things (like resources) may still be uninitialized.
		// Proceed with mild caution.

		LOGGER.info("Hello Fabric world!");

		Registry.register(Registries.DENSITY_FUNCTION_TYPE, new Identifier(MOD_ID, "reader"), Reader.CODEC);
        Registry.register(Registries.BIOME_SOURCE, new Identifier(MOD_ID, "mlcraftbiomesource"), MLCraftBiomeSource.CODEC);

        try {
            Reader.init();
            MLCraftBiomeSource.init();
        } catch (IOException e) {
            throw new RuntimeException(e);
        }

    }

}