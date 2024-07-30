package net.mlcraft.elevationmod;

import com.mojang.serialization.Codec;
import com.mojang.serialization.MapCodec;
import com.mojang.serialization.codecs.RecordCodecBuilder;

import java.io.BufferedInputStream;
import java.io.DataInputStream;
import java.io.FileInputStream;
import java.io.IOException;
import java.util.stream.Stream;

import net.fabricmc.loader.api.FabricLoader;
import net.minecraft.registry.RegistryOps;
import net.minecraft.registry.entry.RegistryEntry;
import net.minecraft.world.biome.Biome;
import net.minecraft.world.biome.BiomeKeys;
import net.minecraft.world.biome.source.TheEndBiomeSource;
import net.minecraft.world.biome.source.util.MultiNoiseUtil;
import net.minecraft.registry.RegistryEntryLookup;
import net.minecraft.world.biome.source.BiomeSource;

public class BiomeReader{

    static MapCodec<BiomeSource> CODEC;

    static BiomeSource BIOME_SOURCE = new BiomeSource() {
        @Override
        protected MapCodec<? extends BiomeSource> getCodec() {
            return BiomeReader.CODEC;
        }

        @Override
        protected Stream<RegistryEntry<Biome>> biomeStream() {
            return Stream.of(meadowBiome);
        }

        @Override
        public RegistryEntry<Biome> getBiome(int x, int y, int z, MultiNoiseUtil.MultiNoiseSampler noise) {
            x = Math.floorMod(x,length);
            z = Math.floorMod(z,length);

            var v = Byte.toUnsignedInt(biomes[x*length+z]);

            return meadowBiome;
        }
    };

    static {
        CODEC = RecordCodecBuilder.mapCodec(instance ->
            instance.group(
                RegistryOps.getEntryCodec(BiomeKeys.SNOWY_TAIGA)
            ).apply(
                instance,
                instance.stable(
                    (meadow) -> {
                        BiomeReader.meadowBiome = meadow;

                        return BIOME_SOURCE;
                    }
                )
            )
        );
    }

    static RegistryEntry<Biome> meadowBiome;

    private static byte[] biomes = null;
    private static int length;

    public static void init() throws IOException {
        var file = FabricLoader.getInstance().getConfigDir().resolve("mlcraft/biome");

        biomes = Helper.readByteFile(file);
        length = (int)Math.sqrt(biomes.length);
    }
}