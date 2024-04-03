package net.mlcraft.elevationmod;

import net.fabricmc.api.ModInitializer;

import net.mlcraft.elevationmod.densityfunction.Lerp;
import net.mlcraft.elevationmod.densityfunction.Distance;
import net.mlcraft.elevationmod.world.gen.MLCraftMapInfo;
import net.mlcraft.elevationmod.world.gen.MLCraftPredicates;
import net.mlcraft.elevationmod.world.gen.NamespacedMapImage;
import net.mlcraft.elevationmod.world.gen.chunk.MLCraftChunkGenerator;

import net.minecraft.server.MinecraftServer;
import net.minecraft.registry.*;
import net.minecraft.util.Identifier;
import net.minecraft.registry.Registry;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;


import net.mlcraft.elevationmod.world.gen.biome.source.MLCraftBiomeSource;


import java.util.HashMap;
import java.util.ArrayList;


public class MLCraft implements ModInitializer {
	public static final String MOD_ID = "mlcraft";
    public static final Logger LOGGER = LoggerFactory.getLogger(MOD_ID);

	public static final int GEN_RADIUS = 8192;
	public static MinecraftServer SERVER;
	public static final RegistryKey<Registry<MLCraftMapInfo>> MLCRAFT_INFO = RegistryKey.ofRegistry(MLCraft.id("worldgen/mlcraft_map_info"));
	public static HashMap<Identifier, NamespacedMapImage> GRAYSCALE_MAPS = new HashMap<>();
	public static HashMap<Identifier, NamespacedMapImage> COLOR_MAPS = new HashMap<>();
	public static Identifier id(String path) {
		return new Identifier(MOD_ID, path);
	}

	@Override
	public void onInitialize() {

		ArrayList<RegistryLoader.Entry<?>> list = new ArrayList<>();
		list.add(new RegistryLoader.Entry<>(MLCRAFT_INFO, MLCraftMapInfo.CODEC));
		list.addAll(RegistryLoader.DYNAMIC_REGISTRIES);
		RegistryLoader.DYNAMIC_REGISTRIES = list;
		BuiltinRegistries.REGISTRY_BUILDER.addRegistry(MLCRAFT_INFO, MLCraftMapInfo::bootstrap);
//		Registry.register(Registries.DENSITY_FUNCTION_TYPE, new Identifier("mlcraft", "distance"), Distance.DISTANCE_CODEC.codec());
//		Registry.register(Registries.DENSITY_FUNCTION_TYPE, new Identifier("mlcraft", "lerp"), Lerp.LERP_CODEC.codec());

		Registry.register(Registries.CHUNK_GENERATOR, id("mlcraft"), MLCraftChunkGenerator.CODEC);
		Registry.register(Registries.BIOME_SOURCE, id("mlcraft"), MLCraftBiomeSource.CODEC);
		MLCraftPredicates.register();
	}

	public static NamespacedMapImage getOrCreateMap(String path, NamespacedMapImage.Type type) {
		if (type == NamespacedMapImage.Type.COLOR) {
			return COLOR_MAPS.computeIfAbsent(new Identifier(path), k -> new NamespacedMapImage(path, NamespacedMapImage.Type.COLOR));
		} else if (type == NamespacedMapImage.Type.GRAYSCALE) {
			return GRAYSCALE_MAPS.computeIfAbsent(new Identifier(path), k -> new NamespacedMapImage(path, NamespacedMapImage.Type.GRAYSCALE));
		} else {
			throw new IllegalArgumentException("tried to create a map with an unknown type!");
		}
	}
	public static NamespacedMapImage getOrCreateMap(Identifier path, NamespacedMapImage.Type type) {
		if (type == NamespacedMapImage.Type.COLOR) {
			return COLOR_MAPS.computeIfAbsent(path, k -> new NamespacedMapImage(path.toString(), NamespacedMapImage.Type.COLOR));
		} else if (type == NamespacedMapImage.Type.GRAYSCALE) {
			return GRAYSCALE_MAPS.computeIfAbsent(path, k -> new NamespacedMapImage(path.toString(), NamespacedMapImage.Type.GRAYSCALE));
		} else {
			throw new IllegalArgumentException("tried to create a map with an unknown type!");
		}
	}
}