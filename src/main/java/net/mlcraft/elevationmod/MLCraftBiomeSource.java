package net.mlcraft.elevationmod;

import com.mojang.serialization.Codec;
import com.mojang.serialization.codecs.RecordCodecBuilder;

import java.io.DataInputStream;
import java.io.FileInputStream;
import java.io.IOException;
import java.util.stream.Stream;

import net.fabricmc.loader.api.FabricLoader;
import net.minecraft.registry.RegistryOps;
import net.minecraft.registry.entry.RegistryEntry;
import net.minecraft.world.biome.Biome;
import net.minecraft.world.biome.BiomeKeys;
import net.minecraft.world.biome.source.util.MultiNoiseUtil;
import net.minecraft.world.biome.source.BiomeSource;

public class MLCraftBiomeSource extends BiomeSource{
    public static final Codec<MLCraftBiomeSource> CODEC = RecordCodecBuilder.create((instance) -> {
        return instance.group(RegistryOps.getEntryCodec(BiomeKeys.DESERT), RegistryOps.getEntryCodec(BiomeKeys.MEADOW)).apply(instance, instance.stable(MLCraftBiomeSource::new));
    });

    private final RegistryEntry<Biome> desertBiome;
    private final RegistryEntry<Biome> meadowBiome;

    public MLCraftBiomeSource(RegistryEntry.Reference<Biome> desertBiome, RegistryEntry.Reference<Biome> meadowBiome) {
        this.desertBiome = desertBiome;
        this.meadowBiome = meadowBiome;
    }

    @Override
    protected Stream<RegistryEntry<Biome>> biomeStream() {
        return Stream.of(this.desertBiome, this.meadowBiome);
    }

    @Override
    protected Codec<? extends BiomeSource> getCodec() {
        return CODEC;
    }

    private static byte[] biomes = null;

    private static int length;

    public static void init() throws IOException {
        var file = FabricLoader.getInstance().getConfigDir().resolve("mlcraft/biome").toFile();

        var d = new DataInputStream(new FileInputStream(file));

        biomes = new byte[d.available()/Byte.BYTES];

        for (int i = 0 ; i < biomes.length; i++){
            biomes[i] = d.readByte();
        }

        length = (int)Math.sqrt(biomes.length);
    }
    @Override
    public RegistryEntry<Biome> getBiome(int x, int y, int z, MultiNoiseUtil.MultiNoiseSampler noise) {
        x = Math.floorMod(x,length);
        z = Math.floorMod(z,length);

        var v = Byte.toUnsignedInt(biomes[x*length+z]);

        if(v == 0) {
            return this.desertBiome;
        }
        else if(v == 1) {
            return this.meadowBiome;
        }
        else {
            throw new RuntimeException();
        }
    }
}